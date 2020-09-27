using System;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;

namespace eMammal_integration_application
{
    public class JsonData
    {
        public Info info { get; set; }
        public Dictionary<string, string> detection_categories { get; set; }
        public Classification_Categories classification_categories { get; set; }
        public List<Image> images { get; set; }

    }

    public class Info
    {
        public string detector { get; set; }
        public string detection_completion_time { get; set; }
        public string format_version { get; set; }
    }

    public class Detection_Categories
    {
        public string _1 { get; set; }
        public string _2 { get; set; }
    }

    public class Classification_Categories
    {
    }

    public class Image
    {
        public Detection[] detections { get; set; }
        public string file { get; set; }
        public dynamic max_detection_conf { get; set; }
    }

    public class Detection
    {
        [JsonProperty(Order = 1)]
        public string category { get; set; }

        [JsonProperty(Order = 2)]
        public dynamic conf { get; set; }

        [JsonProperty(Order = 3)]
        public float[] bbox { get; set; }
    }
}
