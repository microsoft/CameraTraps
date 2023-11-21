using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CameraTrapJsonManagerApp
{
    class SharedFunctions
    {
        public static int GetProgressPercentage(long count, long total)
        {
            return (int)Math.Round((double)(100 * count) / total);
        }
    }
}
