using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.ComponentModel;
using System.IO;
using Newtonsoft.Json;
using System.Runtime.Serialization.Formatters.Binary;
using System.Diagnostics;
using System.Text.RegularExpressions;

namespace CameraTrapJsonManagerApp
{
    /// Creates one or more subsets of a detector API output file (.json), doing either
    /// or both of the following (if both are requested, they happen in this order):
    ///
    /// 1) Retrieve all elements where filenames contain a specified query string, 
    ///    optionally replacing that query with a replacement token. If the query is blank, 
    ///    can also be  used to prepend content to all filenames.
    ///
    /// 2) Create separate .jsons for each unique path, optionally making the filenames 
    ///    in those .json's relative paths.  In this case, you specify an output directory, 
    ///    rather than an output path.  All images in the folder test\foo\bar will end up 
    ///    in a .json file called test_foo_bar.json.

    class SubsetJsonDetectorOutput
    {

        private BackgroundWorker progressReporter;
        private SubsetJsonDetectorOutputOptions options;
        public SubsetJsonDetectorOutput(BackgroundWorker progressReporter, SubsetJsonDetectorOutputOptions options)
        {
            this.progressReporter = progressReporter;
            this.options = options;
        }

        public JsonData SubsetJsonDetectorOutputMain(string inputFileName, string outputFilename,
        SubsetJsonDetectorOutputOptions options, JsonData data = null)
        {

            try
            {
                string progressMsg = string.Empty;

                if (options == null)
                {
                    options = new SubsetJsonDetectorOutputOptions();
                }

                if (options.SplitFolders)
                {
                    if (File.Exists(outputFilename))
                    {
                        progressMsg = "When splitting by folders, output must be a valid directory name";

                        SetStatusMessage(progressMsg);

                        return null;
                    }
                }
                if (data == null)
                {
                    SetLabelProgressMsg("Reading Json file...", ProgressBarStyle.Marquee);

                    data = LoadJson(inputFileName);

                    SetLabelProgressMsg("Loaded Json data...", ProgressBarStyle.Marquee);

                    if (options.DebugMaxImages > 0)
                        data.images = data.images.Take(options.DebugMaxImages).ToList<Image>();

                }
                else
                    data = DeepClone(data);
                if (!string.IsNullOrEmpty(options.Query))
                    data = SubsetJsonDetectorOutputbyQuery(data, options);
                if (options.ConfidenceThreshold != -1)
                    data = SubsetJsonDetectorOutputbyConfidence(data, options);
                if (!options.SplitFolders)
                {
                    SetLabelProgressMsg("Writing to file...", ProgressBarStyle.Marquee);

                    bool result = WriteDetectionResults(data, outputFilename, options);
                    if (!result)
                        return null;

                    SetStatusMessage("File written to " + outputFilename.Replace("/", "\\"));

                    return data;
                }
                else
                {
                    SetLabelProgressMsg("Finding unique folders...", ProgressBarStyle.Marquee);
                    Dictionary<string, List<Image>> foldersToImages = FindUniqueFolders(options, data);
                    if (foldersToImages == null) return null;                    

                    if (options.MakeFolderRelative)
                        foldersToImages = MakeFoldersRelative(foldersToImages);

                    Directory.CreateDirectory(outputFilename);

                    var allImages = data.images;

                    SetLabelProgressMsg("Writing to file...", ProgressBarStyle.Marquee);

                    int count = 0;
                    int totalCount = foldersToImages.Count;

                    // For each folder...
                    foreach (var item in foldersToImages)
                    {

                        count++;
                        string directoryName = item.Key;
                        if (directoryName.Length == 0)
                            directoryName = "base";

                        JsonData dirData = new JsonData();
                        dirData.classification_categories = data.classification_categories;
                        dirData.detection_categories = data.detection_categories;
                        dirData.info = data.info;
                        dirData.images = foldersToImages[directoryName];

                        if (Path.IsPathRooted(directoryName))
                        {
                            string rootPath = Path.GetPathRoot(directoryName);
                            directoryName = directoryName.Replace(rootPath, "");
                        }
                        
                        string jsonFileName = directoryName.Replace('/', '_').Replace('\\', '_') + ".json";
                        
                        if (options.CopyJsonstoFolders)
                            jsonFileName = Path.Combine(outputFilename, directoryName, jsonFileName);
                        else
                            jsonFileName = Path.Combine(outputFilename, jsonFileName);


                        bool result = WriteDetectionResults(dirData, jsonFileName, options);
                        if (!result)
                            return null;

                        SetStatusMessage(string.Format("Wrote {0} images to {1}",
                            dirData.images.Count().ToString(), jsonFileName), count, totalCount);

                    } // ...for each folder

                    data.images = allImages;

                } // if we are/aren't splitting folders

                return data;

            }
            catch (Exception ex)
            {
                ShowError(ex);
                return null;
            }

        } // SubsetJsonDetectorOutputMain

        private JsonData SubsetJsonDetectorOutputbyQuery(JsonData data, SubsetJsonDetectorOutputOptions options)
        {
            // Subset to images whose filename matches options.query; replace all instances of 
            // options.query with options.replacement.
            string progressMsg = string.Empty;

            var imagesIn = data.images;
            List<Image> imagesOut = new List<Image>();

            long count = 0;
            int percentage = 0;
            long totalCount = imagesIn.Count();
            int[] progressPercentagesForDisplay = { 10, 20, 40, 80, 90, 100 };

            progressMsg = string.Format("Subsetting by query {0}, replacement {1} ...", options.Query, options.Replacement);

            SetLabelProgressMsg(progressMsg, ProgressBarStyle.Marquee);

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            foreach (var image in imagesIn)
            {

                count++;

                string file = image.file.ToString();

                if (!string.IsNullOrEmpty(options.Query) && !file.Contains(options.Query))
                    continue;

                if (options.Replacement != null)
                {
                    if (!string.IsNullOrEmpty(options.Query))
                        file = file.Replace(options.Query, options.Replacement);
                    else
                        // If the query is empty and the replacement is non-null, prepend the replacement
                        // to the filename.
                        file = options.Replacement + file;
                }
                percentage = SharedFunctions.GetProgressPercentage(count, totalCount);
                if (progressPercentagesForDisplay.Contains(percentage))
                {
                    progressMsg = string.Format("Subsetting by query {0}, replacement {1} processed {2} of {3}...",
                                options.Query, options.Replacement, count.ToString(), totalCount.ToString());

                    SetLabelProgressPercentageMsg(progressMsg, count, totalCount, ProgressBarStyle.Blocks);
                }

                image.file = file;
                imagesOut.Add(image);

            } // for each image

            data.images = imagesOut;

            SetStatusMessage(string.Format("Finished query search, found {0} matches (of {1})", data.images.Count(), imagesIn.Count()));

            stopwatch.Stop();
            Console.WriteLine(stopwatch.Elapsed.TotalSeconds.ToString());

            return data;

        } // SubsetJsonDetectorOutputbyQuery

        private JsonData SubsetJsonDetectorOutputbyConfidence(JsonData data, SubsetJsonDetectorOutputOptions options)
        {
            if (options.ConfidenceThreshold == -1)
                return data;

            var imagesIn = data.images;
            List<Image> imagesOut = new List<Image>();

            long count = 0;
            int maxChanges = 0;
            long totalCount = imagesIn.Count();

            string progressMsg = string.Empty;

            int[] progressPercentagesForDisplay = { 10, 20, 40, 80, 100 };

            progressMsg = string.Format("Subsetting by confidence >= {0}", options.ConfidenceThreshold.ToString());

            foreach (var item in imagesIn)
            {
                count++;

                int percentage = SharedFunctions.GetProgressPercentage(count, totalCount);
                if (progressPercentagesForDisplay.Contains(percentage))
                    SetLabelProgressMsg(progressMsg, count, totalCount, ProgressBarStyle.Blocks);

                List<Detection> detections = new List<Detection>();

                // detections = [d for d in im['detections'] if d['conf'] >= options.confidence_threshold]

                // Failed images have no detections array, always include them in the output
                if (item.detections == null)
                {
                    imagesOut.Add(item);
                    continue;
                }

                dynamic p;
                dynamic pOrig = item.max_detection_conf;

                // Find all detections above threshold for this image
                foreach (var d in item.detections)
                {
                    dynamic conf = d.conf;
                    if (conf >= options.ConfidenceThreshold)
                        detections.Add(d);
                }

                // If there are no detections above threshold, set the max probability
                // to -1, unless it already had a negative probability.
                if (detections.Count == 0)
                {
                    if (pOrig <= 0)
                        p = pOrig;
                    else
                        p = -1;
                }

                // Otherwise find the maximum confidence
                else
                {
                    p = -1;
                    foreach (var c in detections)
                    {
                        dynamic confidence = c.conf;
                        if (confidence > p)
                            p = confidence;
                    }                    
                }

                item.detections = detections.ToArray();

                // Did this thresholding result in a max-confidence change?
                if (Math.Abs(pOrig - p) > 0.00001)
                {
                    // We should only be *lowering* max confidence values (i.e., making them negative)
                    if (!(pOrig <= 0 || p < pOrig))
                    {
                        string errmsg = string.Format("Confidence changed from {0} to {1}", pOrig, p);

                        // This will get handled by the UI
                        throw new Exception(errmsg);                        
                    }
                    maxChanges += 1;
                }

                item.max_detection_conf = p;
                imagesOut.Add(item);

            } // for each image

            if (imagesOut.Count != imagesIn.Count) { throw new Exception("Image array size mismatch"); }

            data.images = imagesOut;

            SetStatusMessage(string.Format("Finished confidence search, found {0} matches (of {1}), {2} max conf changes", data.images.Count().ToString(),
                              imagesIn.Count().ToString(), maxChanges.ToString()));

            return data;

        } // SubsetJsonDetectorOutputbyConfidence

        private JsonData LoadJson(string inputFileName)
        {
            string json = File.ReadAllText(inputFileName);
            var data = JsonConvert.DeserializeObject<JsonData>(json);
            return data;
        }

        /*
         Based on:

         https://weblog.west-wind.com/posts/2010/Dec/20/Finding-a-Relative-Path-in-NET
          
         Returns a relative path string from a full path based on a base path
         provided.         
        */

        public static string GetRelativePath(string targetPath, string basePath)
        {
            bool useBackslash = (targetPath.Contains("\\") || basePath.Contains("\\"));

            targetPath = targetPath.Replace("/", "\\");
            basePath = basePath.Replace("/", "\\");

            if (!(basePath.EndsWith("\\"))) basePath += "\\";
            
            bool bpRooted = System.IO.Path.IsPathRooted(basePath);
            bool tpRooted = System.IO.Path.IsPathRooted(targetPath);
            if (bpRooted != tpRooted)
            {
                throw new Exception("Can't find relative paths between rooted and non-rooted paths");
            }

            // This is a total hack, but I'll live with it... Uri's can't be instantiated from the full variety 
            // of relative paths we'll see, and the simplest relative-path function in .net operates Uri's.
            String basePathPrefix = null;
            if (!bpRooted) basePathPrefix = "z:\\";
            else if (basePath.StartsWith("\\")) basePathPrefix = "z:";
            if (basePathPrefix != null) basePath = basePathPrefix + basePath;

            String targetPathPrefix = null;
            if (!tpRooted) targetPathPrefix = "z:\\";
            else if (targetPath.StartsWith("\\")) targetPathPrefix = "z:";
            if (targetPathPrefix != null) targetPath = targetPathPrefix + targetPath;

            Uri baseUri = new Uri(basePath, UriKind.Absolute);
            Uri fullUri = new Uri(targetPath, UriKind.Absolute);

            Uri relativeUri = baseUri.MakeRelativeUri(fullUri);

            if (basePathPrefix != null) basePath = basePath.Replace(basePathPrefix, "");
            if (targetPathPrefix != null) targetPath = targetPath.Replace(targetPathPrefix, "");

            // Uri's use forward slashes, possibly convert back
            String toReturn = relativeUri.ToString();
            if (useBackslash) toReturn = toReturn.Replace("/", "\\");
            return toReturn;

            // Test suite for this function; for execution in the interactive console
            if (false)
            {
                String tp;
                String bp;
                String relPath;
                bp = "a/b/c/"; tp = "a/b/c/d/e.jpg"; relPath = GetRelativePath(tp, bp); Console.WriteLine(relPath); Debug.Assert(0 == String.Compare(relPath, "e.jpg"));
                bp = "c:\\a/b/c/"; tp = "c:\\a/b/c/d/e.jpg"; relPath = GetRelativePath(tp, bp); Console.WriteLine(relPath); Debug.Assert(0 == String.Compare(relPath, "e.jpg"));
                bp = "/b"; tp = "/b/c/d/e.jpg"; relPath = GetRelativePath(tp, bp); Console.WriteLine(relPath); Debug.Assert(0 == String.Compare(relPath, "b/c/d/e.jpg"));
                bp = "c:\\a\\b\\c"; tp = "c:\\a\\b\\c.jpg"; relPath = GetRelativePath(tp, bp); Console.WriteLine(relPath);  Debug.Assert(0 == String.Compare(relPath, "..\\c.jpg"));
            }
        }

        private Dictionary<string, List<Image>> MakeFoldersRelative(Dictionary<string, List<Image>> folderstoImages)
        {
            SetLabelProgressMsg(
                string.Format("Converting database-relative paths to individual-json-relative paths...",
                folderstoImages.Count.ToString()), ProgressBarStyle.Marquee);

            foreach (KeyValuePair<string, List<Image>> item in folderstoImages)
            {
                string dirname = item.Key;
                foreach (Image image in item.Value)
                {
                    // Gets the path of [image.file] relative to [dirname]
                    //
                    // Most commonly this is doing something like:
                    //
                    // GetRelativePath("a/b/c/d/e.jpg","a/b/c") == "d/e.jpg"
                    string relativePath = GetRelativePath(image.file, dirname);
                    relativePath = relativePath.Replace("%20", " ");
                    Debug.Assert(!(relativePath.Contains(@"\\")));
                    image.file = relativePath;
                }
            }

            SetLabelProgressMsg("Done",ProgressBarStyle.Marquee);
            return folderstoImages;
        }

        private String[] SplitPath(String p)
        {
            /*
            Splits[path] into all its constituent tokens, e.g.:
    
            c:\blah\boo\goo.txt...becomes:

            ['c:\\', 'blah', 'boo', 'goo.txt']
            */
            return p.Split(new char[] { '\\', '/' }, StringSplitOptions.RemoveEmptyEntries);
        }

        private bool StringEquals(String s1, String s2)
        {
            return (0 == String.Compare(s1, s2));
        }

        private String TopLevelFolder(String path)
        {
            /*
            Gets the top-level folder from the path *path*; on Windows, will use the top-level folder
            that isn't the drive.  E.g., TopLevelFolder(r"c:\blah\foo") returns "c:\blah".  Does not
            include the leaf node, i.e.TopLevelFolder('/blah/foo') returns '/blah'.
            */
            if (path.Length == 0 || path == null)
                return "";

            String[] parts = SplitPath(path);

            if (parts.Length == 1)
                return parts[0];

            // Handle paths like:
            //
            // /, \, /stuff, c:, c:\stuff
            String drive = System.IO.Path.GetPathRoot(path);


            if (StringEquals(parts[0], drive) || StringEquals(parts[0], drive + '/') || StringEquals(parts[0], drive + '\\') || StringEquals(parts[0], "\\") || StringEquals(parts[0], "/"))
                return Path.Combine(parts[0], parts[1]);
            else
                return parts[0];

            // Test suite for this function; for execution in the interactive console
            if (false)
            {
                String p;
                String s;
                p = "blah/foo/bar"; s = TopLevelFolder(p); Console.WriteLine(s); Debug.Assert(StringEquals(s, "blah"));
                p = "/blah/foo/bar"; s = TopLevelFolder(p); Console.WriteLine(s); Debug.Assert(StringEquals(s,"/blah"));
                p = "bar"; s = TopLevelFolder(p); Console.WriteLine(s); Debug.Assert(StringEquals(s,"bar"));
                p = ""; s = TopLevelFolder(p); Console.WriteLine(s); Debug.Assert(StringEquals(s,""));
                p = "c:\\"; s = TopLevelFolder(p); Console.WriteLine(s); Debug.Assert(StringEquals(s,"c:\\"));
                p = @"c:\blah"; s = TopLevelFolder(p); Console.WriteLine(s); Debug.Assert(StringEquals(s,"c:\\blah"));
                p = @"c:\foo"; s = TopLevelFolder(p); Console.WriteLine(s); Debug.Assert(StringEquals(s,"c:\\foo"));
                p = "c:/foo"; s = TopLevelFolder(p); Console.WriteLine(s); Debug.Assert(StringEquals(s,"c:/foo"));
                p = @"c:\foo/bar"; s = TopLevelFolder(p); Console.WriteLine(s); Debug.Assert(StringEquals(s,"c:\\foo"));
            }
       }

        private Dictionary<string, List<Image>> FindUniqueFolders(SubsetJsonDetectorOutputOptions options, JsonData data)
        {
            Dictionary<string, List<Image>> folderstoImages = new Dictionary<string, List<Image>>();

            foreach (var imagedata in data.images)
            {

                string filePath = imagedata.file.ToString();
                string directoryName = string.Empty;
                List<Image> imageList = new List<Image>();

                if (options.SplitFolderMode.ToLower() == "bottom")
                    directoryName = Path.GetDirectoryName(filePath);

                else if (options.SplitFolderMode.ToLower() == "nfrombottom")
                {
                    directoryName = Path.GetDirectoryName(filePath);
                    for (int n = 0; n < options.nDirectoryParam; n++)
                    {
                        if (directoryName.Length == 0)
                        {
                            string msg = string.Format("Error: cannot walk {0} folders from the bottom in path {1}",
                                options.nDirectoryParam, filePath);
                            SetStatusMessage(msg);
                            return null;
                        }
                        directoryName = Path.GetDirectoryName(directoryName);
                    }
                }
                else if (options.SplitFolderMode.ToLower() == "nfromtop")
                {
                    directoryName = Path.GetDirectoryName(filePath);

                    // Split string into folders, keeping delimiters
                    String delimiter = @"([\\/])";
                    String[] tokens = Regex.Split(directoryName, delimiter);
                    int nTokensToKeep = ((options.nDirectoryParam + 1) * 2) - 1;
                    if (nTokensToKeep > tokens.Length)
                    {
                        string msg = string.Format("Error: cannot walk {0} folders from the top in path {1}",
                                options.nDirectoryParam, filePath);
                        SetStatusMessage(msg);
                        return null;
                    }
                    tokens = tokens.Take(nTokensToKeep).ToArray();
                    directoryName = String.Join("",tokens);
                }
                else if (options.SplitFolderMode.ToLower() == "top")
                {
                    directoryName = TopLevelFolder(filePath);
                }

                if (!folderstoImages.ContainsKey(directoryName))
                {
                    imageList.Add(imagedata);
                    folderstoImages.Add(directoryName, imageList);
                }
                else
                {
                    var currentImgArray = folderstoImages[directoryName];
                    currentImgArray.Add(imagedata);
                    folderstoImages[directoryName] = currentImgArray;
                }

            } // ...for each image

            SetLabelProgressMsg(string.Format("Found {0} unique folders", folderstoImages.Count.ToString(),
               folderstoImages.Count.ToString()), ProgressBarStyle.Marquee);
           
            return folderstoImages;
        }

        private bool WriteDetectionResults(JsonData data, string outputFileName, SubsetJsonDetectorOutputOptions options)
        {
            // Convert all paths to forward slashes unless they're absolute
            if (options.UseForwardSlashesWhenPossible)
            {
                foreach (Image im in data.images)
                {
                    if ((im.file.Contains("\\")) && (!(im.file.Contains("/"))) && (!Path.IsPathRooted(im.file)))
                    {
                        im.file = im.file.Replace('\\', '/');
                    }
                }
            }

            // Write the detector ouput *data* to *output_filename*
            if (!Path.IsPathRooted(outputFileName))
            {
                string msg = string.Format("Must specify an absolute output path");
                SetStatusMessage(msg);
                return false;
            }

            if (!options.OverwriteJsonFiles && File.Exists(outputFileName))
            {
                string msg = string.Format("File {0} exists", outputFileName);
                SetStatusMessage(msg);               
                return false;
            }

            outputFileName = outputFileName.Replace("/", "\\");
            string directoryPath = outputFileName.Substring(0, outputFileName.LastIndexOf("\\"));

            if (options.CopyJsonstoFolders && options.CopyJsonstoFoldersDirectoriesMustExist)
            {
                if (!Directory.Exists(directoryPath))
                {
                    string msg = String.Format("Directory {0} does not exist", outputFileName);
                    SetStatusMessage(msg);                  
                    return false;
                }
            }
            else
            {
              Directory.CreateDirectory(directoryPath);
            }
            try
            {
                String ext = System.IO.Path.GetExtension(outputFileName);
                if (!(ext.Equals(".json")))
                {
                    string msg = "Output file name must end in .json";
                    SetStatusMessage(msg);
                    return false;
                }

                using (FileStream fs = File.Create(outputFileName))
                using (StreamWriter sw = new StreamWriter(fs))
                using (JsonTextWriter jtw = new JsonTextWriter(sw)
                {
                    Formatting = Formatting.Indented,
                    Indentation = 1,
                    IndentChar = ' ',
                })
                {
                    JsonSerializer _serializer = new JsonSerializer
                    {
                        NullValueHandling = NullValueHandling.Ignore
                    };
                    _serializer.Serialize(jtw, data);
                }
            }
            catch (Exception ex)
            {
                SetStatusMessage(ex.Message);
            }
            return true;

        } // WriteDetectionResults()

        private static T DeepClone<T>(T obj)
        {
            using (var ms = new MemoryStream())
            {
                var formatter = new BinaryFormatter();
                formatter.Serialize(ms, obj);
                ms.Position = 0;
                return (T)formatter.Deserialize(ms);
            }
        }

        private void ShowError(Exception ex)
        {
            progressReporter.ReportProgress(0, new ProgressDetails
            {
                ShowProgressBar = false,
                SetStatusTextBoxMessage = true,
                Message = ex.ToString()
            });
        }

        private void SetStatusMessage(string message)
        {
            progressReporter.ReportProgress(0,new ProgressDetails 
                { SetStatusTextBoxMessage = true, Message = message });
        }

        private void SetStatusMessage(string message, int count, int totalCount)
        {
            progressReporter.ReportProgress(count, new ProgressDetails
            {
                Maximum = totalCount,
                CurrentCount = count,
                SetStatusTextBoxMessage = true,
                ShowProgressBar = true,
                Message = message
            });
        }

        private void SetLabelProgressMsg(string message, ProgressBarStyle style)
        {
            progressReporter.ReportProgress(0, new ProgressDetails
            {
                SetlabelProgressMessage = true,
                Message = message,
                ShowProgressBar = true,
                style = style
            });
        }

        private void SetLabelProgressMsg(string message, long count, long totalCount, ProgressBarStyle style)
        {
            progressReporter.ReportProgress(0, new ProgressDetails
            {
                SetlabelProgressMessage = true,
                ShowProgressBar = true,
                style = style,
                CurrentCount = count,
                Maximum = totalCount,
                Message = message
            });
        }
        private void SetLabelProgressPercentageMsg(string message, long count, long totalCount, ProgressBarStyle style)
        {
            progressReporter.ReportProgress(0, new ProgressDetails
            {
                SetlabelProgressPercentageMessage = true,
                ShowProgressBar = true,
                style = style,
                CurrentCount = count,
                Maximum = totalCount,
                Message = message
            });
        }

    } // class SubsetJsonDetectorOutput

} // namespace CameraTrapJsonManagerApp

