using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Newtonsoft.Json.Linq;
using System.Runtime.Serialization.Formatters.Binary;

namespace CameraTrapJsonManagerApp
{
    /// <summary>
    /// Helper class
    /// </summary>
    class SubsetJsonDetectorOutputOptions
    {
        
        // Only process files containing the token 'query'
        public string Query { get; set; } = null;

        // Replace 'query' with 'replacement' if 'replacement' is not None.  If 'query' is None,
        // prepend 'replacement'
        public string Replacement { get; set; } = null;

        // Should we split output into individual .json files for each folder?
        public bool SplitFolders { get; set; } = false;

        // Folder level to use for splitting ("top", "bottom", or "n_from_bottom")
        public string SplitFolderMode { get; set; } = "bottom";

        // When using the 'n_from_bottom' parameter to define folder splitting, this
        // defines the number of directories from the bottom.  'n_from_bottom' with
        // a parameter of zero is the same as 'bottom'.
        public int nDirectoryParam { get; set; } = 0;

        // Only meaningful if split_folders is True: should we convert pathnames to be relative
        // the folder for each .json file?
        public bool MakeFolderRelative { get; set; } = false;

        // Only meaningful if split_folders and make_folder_relative are True: if not None, 
        // will copy .json files to their corresponding output directories, relative to 
        // output_filename
        public bool CopyJsonstoFolders { get; set; } = false;

        // Should we over-write .json files?
        public bool OverwriteJsonFiles { get; set; } = false;

        // If copy_jsons_to_folders is true, do we require that directories already exist?
        public bool CopyJsonstoFoldersDirectoriesMustExist { get; set; } = true;

        // Threshold on confidence
        public double ConfidenceThreshold { get; set; } = -1;
        public int DebugMaxImages { get; set; }

        // Not exposed through the UI
        public bool UseForwardSlashesWhenPossible { get; set; } = true;
    }
}
