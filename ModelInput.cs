using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearning_ImageClassification_Binary
{
    public class ModelInput
    {
        public byte[]? Image { get; set; }

        public UInt32 LabelAsKey { get; set; }

        public string? ImagePath { get; set; }

        public string? Label { get; set; }
    }
}
