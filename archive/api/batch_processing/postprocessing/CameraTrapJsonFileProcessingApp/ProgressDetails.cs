using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CameraTrapJsonManagerApp
{
    /// <summary>
    /// Class used for displaying program status information
    /// </summary>
    class ProgressDetails
    {
        public ProgressBarStyle style { get; set; }
        public bool ShowProgressBar { get; set; } = false;
        public bool SetlabelProgressMessage { get; set; } = false;
        public bool SetlabelProgressPercentageMessage { get; set; } = false;
        public bool SetStatusTextBoxMessage { get; set; } = false;
        public string Message { get; set; } = string.Empty;
        public long Maximum { get; set; } = 0;
        public long CurrentCount { get; set; } = 0;
        public bool RemoveProgressInfo { get; set; } = false;
        public bool EnableControls { get; set; } = false;

        public bool IsError { get; set; }

    }
}
