import msgParser
import carState
import carControl
import joblib
import numpy as np
import tensorflow as tf

class Driver(object):
    '''Driver for SCRC using a trained, optimized neural network.''' 

    def __init__(self, stage):
        # State and control
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()

        # Load model & scaler
        model_path = "../trained_model.h5"
        scaler_path = "../scaler_input.pkl"
        self.model = tf.keras.models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)

        # Normalization constants
        self.scale_min = tf.constant(scaler.data_min_, dtype=tf.float32)
        self.scale_range = tf.constant(scaler.data_max_ - scaler.data_min_, dtype=tf.float32)

        # Input buffer
        self.input_dim = self.model.input_shape[1]
        self.buf = np.zeros((1, self.input_dim), dtype=np.float32)

        # Gear logic parameters
        self.MIN_GEAR = 1
        self.MAX_GEAR = 6
        self.gear_up_rpm   = [5000, 6000, 7000, 7000, 7000, 0]
        self.gear_down_rpm = [0, 3000, 3300, 5000, 5500, 6000]

        # Reverse/forward state machine
        self.rev_state = 0        # 0=normal,1=reversing,2=forward_lock
        self.rev_count = 0
        self.fwd_count = 0
        self.REV_DURATION = 60     # reversing steps (short)
        self.FWD_DURATION = 180    # forward lock steps (longer)

        # Inference function
        @tf.function
        def predict_fn(x):
            x_norm = (x - self.scale_min) / self.scale_range
            preds = self.model(x_norm, training=False)  # (1,5)
            return preds[0,0], preds[0,1], preds[0,2], preds[0,3], preds[0,4]
        self.predict_fn = predict_fn

    def init(self):
        angles = [0]*19
        for i in range(5): angles[i] = -90+15*i; angles[18-i] = 90-15*i
        for i in range(5,9): angles[i] = -20+5*(i-5); angles[18-i] = 20-5*(i-5)
        return self.parser.stringify({'init': angles})

    def drive(self, msg):
        s = self.state
        c = self.control
        s.setFromMsg(msg)

        # Warm-up
        if s.getCurLapTime() < 0:
            c.setAccel(1.0); c.setGear(self.MIN_GEAR)
            return c.toMsg()

        # Buffer fill
        idx=0
        self.buf[0,idx]=s.angle; idx+=1
        for v in s.opponents: self.buf[0,idx]=v; idx+=1
        for attr in ('rpm','speedX','speedY','speedZ','trackPos','z'):
            self.buf[0,idx]=getattr(s,attr); idx+=1
        self.buf[0,idx]=float(s.gear); idx+=1
        for v in s.track: self.buf[0,idx]=v; idx+=1
        for v in s.wheelSpinVel: self.buf[0,idx]=v; idx+=1

        # Inference
        c_val,b_val,s_val,a_val,g_float = self.predict_fn(tf.constant(self.buf))
        clutch,brake,steer,accel = map(lambda t: float(t.numpy()), (c_val,b_val,s_val,a_val))

        # Determine gear
        gear = s.getGear()
        rpm  = s.getRpm()

        # State machine
        if self.rev_state == 0:
            # normal shifting
            if gear < self.MAX_GEAR and rpm >= self.gear_up_rpm[gear-1]:
                gear+=1; self.rev_count=0
            elif gear > self.MIN_GEAR and rpm <= self.gear_down_rpm[gear-1]:
                gear-=1; self.rev_count=0
            # trigger reverse on stall
            if int(s.getDistRaced())>2 and s.speedX<4:
                self.rev_state=1; self.rev_count=0
        elif self.rev_state == 1:
            # reversing
            gear = -1
            self.rev_count+=1
            if self.rev_count >= self.REV_DURATION:
                self.rev_state=2; self.fwd_count=0
                gear=self.MIN_GEAR
        else:  # forward_lock
            # force forward gear
            gear = max(self.MIN_GEAR, gear)
            self.fwd_count+=1
            if self.fwd_count >= self.FWD_DURATION:
                self.rev_state=0
                self.rev_count=self.fwd_count=0

        # Apply controls
        c.setClutch(np.clip(clutch,0,1))
        c.setBrake(np.clip(brake,0,1))
        c.setSteer(np.clip(steer,-1,1))
        c.setAccel(np.clip(accel,0,1))
        c.setGear(int(gear))
        return c.toMsg()

    def onShutDown(self): pass
    def onRestart(self): pass
