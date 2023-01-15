using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace CameraTrapJsonManagerApp
{
    public class DetectorMetadata
    {
        public string megadetector_version { get; set; }
        public float typical_detection_threshold { get; set; }
        public float conservative_detection_threshold { get; set; }
    }

    public class ClassifierMetadata
    {
        public float typical_classification_threshold { get; set; }
    }

    public class Info
    {
        public string detection_completion_time { get; set; }
        public string format_version { get; set; }
        public string detector { get; set; }
        public DetectorMetadata detector_metadata { get; set; }
        public string classifier { get; set; }
        public string classification_completion_time { get; set; }
        public ClassifierMetadata classifier_metadata { get; set; }
    }

    public class Detection
    {
        [JsonProperty(Order = 1)]
        public string category { get; set; }

        [JsonProperty(Order = 2)]
        public dynamic conf { get; set; }

        [JsonProperty(Order = 3)]
        public float[] bbox { get; set; }
        public List<dynamic> classifications { get; set; }
    }

    public class Image
    {
        public string file { get; set; }
        public dynamic max_detection_conf { get; set; }
        public Detection[] detections { get; set; }
        public string failure { get; set; }        
    }

    public class JsonData
    {
        public Info info { get; set; }
        public Dictionary<string, string> detection_categories { get; set; }
        public Dictionary<string, string> classification_categories { get; set; }
        public List<Image> images { get; set; }
    }
    
}
