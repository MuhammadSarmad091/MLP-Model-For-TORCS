import msgParser
import carState
import carControl
import joblib
import numpy as np
import tensorflow as tf

class Driver(object):
    '''Driver for SCRC using a trained, optimized neural network.''' 

    def __init__(self, stage):
        # Initialize state
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()

        # Load model & artifacts (no compile for speed)
        model_path = "../trained_model.h5"
        scaler_path = "../scaler_input.pkl"
        self.model = tf.keras.models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)

        # Prepare scaler constants for graph-based normalization
        self.scale_min = tf.constant(scaler.data_min_, dtype=tf.float32)
        self.scale_range = tf.constant(scaler.data_max_ - scaler.data_min_, dtype=tf.float32)

        # Pre-allocate input buffer (1, D)
        self.input_dim = self.model.input_shape[1]
        self.buf = np.zeros((1, self.input_dim), dtype=np.float32)

        # Build a tf.function for end-to-end inference
        @tf.function
        def predict_fn(x):
            # x: tensor shape (1,D), raw features
            x_norm = (x - self.scale_min) / self.scale_range
            preds = self.model(x_norm, training=False)  # shape (1,5)
            # unpack predictions from single tensor
            c = preds[0, 0]
            b = preds[0, 1]
            s = preds[0, 2]
            a = preds[0, 3]
            g_float = preds[0, 4]  # continuous gear prediction
            return c, b, s, a, g_float

        self.predict_fn = predict_fn

    def init(self):
        # Rangefinder angles
        angles = [0] * 19
        for i in range(5):
            angles[i] = -90 + 15 * i
            angles[18-i] = 90 - 15 * i
        for i in range(5,9):
            angles[i] = -20 + 5*(i-5)
            angles[18-i] = 20 - 5*(i-5)
        return self.parser.stringify({'init': angles})

    def drive(self, msg):
        # Parse sensors
        self.state.setFromMsg(msg)

        if self.state.getCurLapTime() < 0:
            self.control.setAccel(1)
            self.control.setGear(1)
            return self.control.toMsg()

        # Fill buffer in fixed order without new allocations
        idx = 0
        # angle
        self.buf[0, idx] = self.state.angle; idx += 1
        # opponents
        for v in self.state.opponents:
            self.buf[0, idx] = v; idx += 1
        # core dynamics
        for attr in ('rpm','speedX','speedY','speedZ','trackPos','z'):
            self.buf[0, idx] = getattr(self.state, attr); idx += 1
        # include current gear as input feature (lowercase 'gear')
        self.buf[0, idx] = float(self.state.gear); idx += 1
        # track sensors
        for v in self.state.track:
            self.buf[0, idx] = v; idx += 1
        # wheel spin velocities
        for v in self.state.wheelSpinVel:
            self.buf[0, idx] = v; idx += 1

        # Call optimized graph
        c, b, s, a, g_float = self.predict_fn(tf.constant(self.buf))

        # Convert to Python values
        clutch = float(c.numpy())
        brake = float(b.numpy())
        steer = float(s.numpy())
        accel = float(a.numpy())
        # Use ceiling to convert floating gear to integer

        gear = self.state.getGear()
        if g_float < 0:
            gear = -1
        elif np.abs(gear - g_float) >= 1.0:
            gear = int(g_float.numpy())


        # Set controls
        self.control.setClutch(np.clip(clutch, 0.0, 1.0))
        self.control.setBrake(np.clip(brake, 0.0, 1.0))
        self.control.setSteer(np.clip(steer, -1.0, 1.0))
        self.control.setAccel(np.clip(accel, 0.0, 1.0))
        self.control.setGear(gear)

        return self.control.toMsg()

    def onShutDown(self):
        pass

    def onRestart(self):
        pass
