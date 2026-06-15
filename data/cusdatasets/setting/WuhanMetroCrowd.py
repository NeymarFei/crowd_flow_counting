try:
    from easydict import EasyDict as edict
except ImportError:
    class edict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value


__C_WUHAN = edict()

cfg_data = __C_WUHAN

__C_WUHAN.TRAIN_SIZE = (576, 720)
__C_WUHAN.TRAINING_MAX_LONG = 1280
__C_WUHAN.TRAINING_MAX_SHORT = 720
__C_WUHAN.TEST_MAX_LONG = 1280
__C_WUHAN.TEST_MAX_SHORT = 720
__C_WUHAN.DATA_PATH = './data/WuhanMetroCrowd_SDNet'
__C_WUHAN.TRAIN_LST = 'train.txt'
__C_WUHAN.VAL_LST = 'val.txt'
__C_WUHAN.TEST_LST = 'test.txt'

__C_WUHAN.MEAN_STD = (
    [117 / 255., 110 / 255., 105 / 255.],
    [67.10 / 255., 65.45 / 255., 66.23 / 255.]
)

__C_WUHAN.DEN_FACTOR = 200.

__C_WUHAN.TRAIN_BATCH_SIZE = 1
__C_WUHAN.TRAIN_FRAME_INTERVALS = (1, 4)
__C_WUHAN.VAL_FRAME_INTERVALS = 4
__C_WUHAN.VAL_BATCH_SIZE = 1
