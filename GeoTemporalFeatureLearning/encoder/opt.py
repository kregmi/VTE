# file paths are completed; suffixes are added in the dataloader: either for train or val.
# phase = 'train'


class Arguments_Data_SF(object):
    def __init__(self, phase):
        super(Arguments_Data_SF, self).__init__()
        self.phase = phase

        self.pickle_file = './BDD/data_prep/bdd_annotations_pickle_' + str(self.phase) + '.pkl'
        self.label_json = './BDD/bdd100k/labels/bdd100k_labels_images_'  + str(self.phase) + '.json'

        self.gsv_dir = './Dataset/GSV/'  + str(self.phase)
        self.bdd_dir = './Dataset/BDD/'  + str(self.phase)
        self.bdd_gsv_filename_mapper='./annotated_data_' + self.phase + '_SF.txt'
        self.gps_limit = [37.65, 37.81, -122.5, -122.38]

        self.batch_size = 2
        self.num_frames = 8
        self.test_num_frames= 30
        self.test_batch_size = 1
        self.timesteps=30



class Arguments_Data_NY(object):
    def __init__(self, phase):
        super(Arguments_Data_NY, self).__init__()
        self.phase = phase

        self.pickle_file = './BDD/data_prep/bdd_annotations_pickle_' + str(self.phase) + '.pkl'
        self.label_json = './BDD/bdd100k/labels/bdd100k_labels_images_'  + str(self.phase) + '.json'

        self.gsv_dir = './Dataset/GSV/'  + str(self.phase)
        self.bdd_dir = './Dataset/BDD/'  + str(self.phase)
        self.bdd_gsv_filename_mapper = './annotated_data_val_NY.txt'
        self.gps_limit = [40.7073, 40.7381, -74.01486, -73.9687]

        self.batch_size = 2
        self.num_frames = 8
        self.test_num_frames = 30
        self.test_batch_size = 1
        self.timesteps = 30


class Arguments_Data_Berkeley(object):
    def __init__(self, phase):
        super(Arguments_Data_Berkeley, self).__init__()
        self.phase = phase

        self.pickle_file = './BDD/data_prep/bdd_annotations_pickle_' + str(self.phase) + '.pkl'
        self.label_json = './BDD/bdd100k/labels/bdd100k_labels_images_'  + str(self.phase) + '.json'
        self.gsv_dir = './Dataset/GSV/'  + str(self.phase)
        self.bdd_dir = './Dataset/BDD/'  + str(self.phase)

        self.gps_limit = [37.72409913, 37.897474, -122.312608, -122.100853]
        self.bdd_gsv_filename_mapper = './annotated_data_val_Berkeley.txt'

        self.batch_size = 2
        self.num_frames = 8
        self.test_num_frames= 30
        self.test_batch_size = 1
        self.timesteps=30



class Arguments_Data_BayArea(object):
    def __init__(self, phase):
        super(Arguments_Data_BayArea, self).__init__()
        self.phase = phase

        self.pickle_file = './BDD/data_prep/bdd_annotations_pickle_' + str(self.phase) + '.pkl'
        self.label_json = './BDD/bdd100k/labels/bdd100k_labels_images_'  + str(self.phase) + '.json'
        self.gsv_dir = './Dataset/GSV/'  + str(self.phase)
        self.bdd_dir = './Dataset/BDD/'  + str(self.phase)

        self.gps_limit = [37.419279, 37.507089, -122.258048, -122.1054]
        self.bdd_gsv_filename_mapper = './annotated_data_val_Bay_Area.txt'

        self.batch_size = 2
        self.num_frames = 8
        self.test_num_frames= 30
        self.test_batch_size = 1
        self.timesteps=30


class Arguments_Model(object):
    def __init__(self):
        super(Arguments_Model, self).__init__()
        self.lr = 0.0001 # learning rate
        self.validate=False
        self.start_epoch=0
        self.epochs=100
        self.print_every=10
        self.snapshot_pref = ""
        self.arch='resnet18'

