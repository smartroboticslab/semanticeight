class TestCase:

    def __init__(self):
        self.sequence_name    = None
        self.sensor_type      = None
        self.voxel_impl       = None
        self.map_res          = None
        self.downsampling_factor      = None
        self.config_yaml_path = None

    def toString(self):
        return self.sequence_name + " | " + self.sensor_type + " | " + self.voxel_impl + \
               " | map res: " + str(self.map_res) + " | downsampling-factor: " + str(self.downsampling_factor)


