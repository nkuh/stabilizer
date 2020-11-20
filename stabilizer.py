import json
import asyncio
from collections import OrderedDict as OD
import logging

import numpy as np
from scipy import signal

logger = logging.getLogger()


def compare(a,b):
    a = -bilinear_transform(a)/(2*np.pi)
    b = -bilinear_transform(b)/(2*np.pi)
    diff = abs(a-b)
    if diff > 1:
        print("Warning")
    if min(abs(a), abs(b) != 0.):
        ratio = max(a, b)/min(a, b)
        if (ratio-1) > 0.01:
            print("Warning")
    
def bilinear_transform(z):
    T = IIR.t_update
    return (2/T)*(z-1.)/(z+1.)

def bilinear_inverse(s):
    T = IIR.t_update
    return (1+T*s/2)/(1-T*s/2)   


class StabilizerError(Exception):
    pass


class StabilizerConfig:
    async def connect(self, host, port=1235):
        self.reader, self.writer = await asyncio.open_connection(host, port)

    async def set(self, channel, cascade, iir):
        value = OD([("channel", channel), ("iir", iir.as_dict())])
        if cascade != 0:
            channel = "_{}{}".format(chr(ord('a')+cascade),channel)
        request = {
            "req": "Write",
            "attribute": "stabilizer/iir{}/state".format(channel),
            "value": json.dumps(value, separators=[',', ':']).replace('"', "'"),
        }
        s = json.dumps(request, separators=[',', ':'])
        assert "\n" not in s
        logger.debug("send %s", s)
        self.writer.write(s.encode("ascii") + b"\n")
        r = (await self.reader.readline()).decode()
        logger.debug("recv %s", r)
        ret = json.loads(r, object_pairs_hook=OD)
        if ret["code"] != 200:
            raise StabilizerError(ret)
        return ret


class IIR:
    t_update = 2e-6
    full_scale = float((1 << 15) - 1)

    def __init__(self):
        self.ba = np.zeros(5, np.float32)
        self.y_offset = 0.
        self.y_min = -self.full_scale - 1
        self.y_max = self.full_scale

    def as_dict(self):
        iir = OD()
        iir["ba"] = [float(_) for _ in self.ba]
        iir["y_offset"] = self.y_offset
        iir["y_min"] = self.y_min
        iir["y_max"] = self.y_max
        return iir

    def configure_pi(self, kp, ki, g=0.):
        ki = np.copysign(ki, kp)*self.t_update*2
        g = np.copysign(g, kp)
        eps = np.finfo(np.float32).eps
        if abs(ki) < eps:
            a1, b0, b1 = 0., kp, 0.
        else:
            if abs(g) < eps:
                c = 1.
            else:
                c = 1./(1. + ki/g)
            a1 = 2*c - 1.
            b0 = ki*c + kp
            b1 = ki*c - a1*kp
            if abs(b0 + b1) < eps:
                raise ValueError("low integrator gain and/or gain limit")
        self.ba[0] = b0
        self.ba[1] = b1
        self.ba[2] = 0.
        self.ba[3] = a1
        self.ba[4] = 0.
        
    def configure_pii(self, kp=1., f_i=1000., f_ii=80.01, f_ilim=80.):
        self.z = [bilinear_inverse(-2*np.pi*f_ii), bilinear_inverse(-2*np.pi*f_i)]
        self.p = [bilinear_inverse(-2*np.pi*0), bilinear_inverse(-2*np.pi*f_ilim)]
        self.k = kp
        self.zpk = signal.ZerosPolesGain(self.z, self.p, self.k, dt=self.t_update)
        
        # turn into TF to get b's and a's
        self.tf = self.zpk.to_tf()
        # round to f32        
        self.ba[0:3] = np.float32(self.tf.num[0:3])
        self.ba[3] = np.float32(-self.tf.den[1])
        self.ba[4] = np.float32(-self.tf.den[2])
        # get rounded zpk
        self.zpk_f32 = signal.TransferFunction([self.ba[0],self.ba[1],self.ba[2]],[np.float32(1.),-self.ba[3],-self.ba[4]],dt=self.t_update).to_zpk()
    
    def check(self):
        #if self.zpk approx. equal to self.zpk_f32 -> everything fine 
        compare( self.zpk.zeros[0],self.zpk_f32.zeros[0])
        compare( self.zpk.zeros[1],self.zpk_f32.zeros[1])
        compare( self.zpk.poles[0],self.zpk_f32.poles[0])
        compare( self.zpk.poles[1],self.zpk_f32.poles[1])
        compare( self.zpk.gain,self.zpk_f32.gain)

    def configure_pd(self, kp=1., f_d=12000., g=20.):
        Ts = self.t_update
        pi = np.pi
        f_dtilde = pi*f_d*Ts
        b2 = 0
        b1 = np.float32(-kp*((1-f_dtilde)/(1/g+f_dtilde)))
        b0 = np.float32(kp*((1+f_dtilde)/(1/g+f_dtilde)))
        a2 = 0
        a1 = np.float32(-(1/g-f_dtilde)/(1/g+f_dtilde))

        kp_eff = (b0+b1)/(1+a1)
        f_d_eff = (b0+b1)/((b0-b1)*pi*Ts)
        g_eff = (1+a1)*(b0-b1)/((1-a1)*(b0+b1))
        error = max(abs((kp_eff/kp)-1), abs((f_d_eff/f_d)-1), abs((g_eff/g)-1))
        if error > 0.01:
            raise Warning("realizable transfer function differs")

        self.ba[0] = b0
        self.ba[1] = b1
        self.ba[2] = b2
        self.ba[3] = -a1
        self.ba[4] = -a2

    def set_x_offset(self, o):
        b = self.ba[:3].sum()*self.full_scale
        self.y_offset = b*o


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--stabilizer", default="10.0.16.99")
    p.add_argument("-c", "--channel", default=0, type=int,
                   help="Stabilizer channel to configure")
    p.add_argument("-o", "--offset", default=0., type=float,
                   help="input offset, in units of full scale")
    p.add_argument("-p", "--proportional-gain", default=1., type=float,
                   help="Proportional gain, in units of 1")
    p.add_argument("-i", "--integral-gain", default=0., type=float,
                   help="Integral gain, in units of Hz, "
                        "sign taken from proportional-gain")
    p.add_argument("-k", "--cascade", default=0, type=int, 
                    help="Cascade# of selected channel")

    args = p.parse_args()

    loop = asyncio.get_event_loop()
    # loop.set_debug(True)
    logging.basicConfig(level=logging.DEBUG)

    async def main():
        i = IIR()
        #i.configure_pi(args.proportional_gain, args.integral_gain)
        #i.configure_pd()
        i.configure_pii()
        i.check()
        i.set_x_offset(args.offset)
        s = StabilizerConfig()
        await s.connect(args.stabilizer)
        assert args.channel in range(2)
        assert args.cascade in range(2)
        r = await s.set(args.channel, args.cascade, i)

    loop.run_until_complete(main())