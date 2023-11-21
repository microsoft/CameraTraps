using NLog;
using System.Data;
using System.Linq;
using System.Text;
using System.Windows;

namespace eMammal_integration_application
{
    public class eMammalIntegration
    {

        eMammalIntegrationWindow window;

        Logger logger = LogManager.GetCurrentClassLogger();

        // Category constants
        const string animal = "1";
        const string person = "2";
        const string vehicle = "3";

        eMammalMySQLOps db;
        public eMammalIntegration(eMammalIntegrationWindow window)
        {
            this.window = window;
            db = new eMammalMySQLOps(window);
        }

        /// <summary>
        /// Create list of image sequence and annotations for bulk insertion 
        /// Call function to insert or update annotations
        /// Update progress bar and message
        /// </summary>
        /// <param name="data"></param>
        /// <param name="deploymentId"></param>
        /// <param name="eMammalCategory"></param>
        public bool ProcessDetections(JsonData data, int deploymentId, string deploymentName, Category eMammalCategory)
        {

            StringBuilder logImages = new StringBuilder();

            int totalImages = data.images.Count();
            window.ProgressbarUpdateProgress.Maximum = totalImages;

            logger.Info(Constants.LOG_GETTING_IMAGE_SEQUENCE_DATA_FROM_DB);
            Common.ShowProgress(window, Constants.PROGRESS_GETTING_IMAGE_SEQUENCE_DATA_FROM_DB, 1);

            DataTable dtImageSequences = db.GetsequenceIDsfromDB(deploymentId);
            int imageSequenceCount = -1;
            if (dtImageSequences == null)
            {
                logger.Info(Constants.LOG_COULD_NOT_RETRIEVE_IMAGE_SEQUENCES_FROM_DATABASE);
                return false;
            }
            else
                imageSequenceCount = dtImageSequences.Rows.Count;

            logger.Info(Constants.LOG_NUM_IMAGE_SEQUENCES + "  " + dtImageSequences.Rows.Count.ToString());

            if (imageSequenceCount == 0)
            {
                string msg = string.Format("The selected eMammal deployment {0} does not contain any images", deploymentName);
                logger.Info(msg);

                Common.SetMessage(window,msg,true);
                return false;
            }

            int showProgressCount = 10;
            showProgressCount = Common.GetShowProgressCount(showProgressCount, totalImages);

            logger.Info(Constants.LOG_ITERATING_IMAGES_IN_JSON_FILE);

            Common.ShowProgress(window, Constants.PROCESSING_IMAGES, 1);

            // This variable will be set to true if there is atleast one matching image that matches the image (by name) 
            // in eMammal database that is in the 
            bool foundImage = false;
            int logCount = 0;
            int maxBulkInsertCount = 10000;
            int count = 0;
            int progressCount = 1;

            bool recordsAdded = false;
            bool imageNotFoundProgressSet = false;

            StringBuilder sql = db.GetBulkInsertInitialString();
            foreach (var image in data.images)
            {
                recordsAdded = false;

                string filePath = image.file.Replace("/", "\\");
                string imageName = System.IO.Path.GetFileName(filePath);
                string imageplusLastFolderName = "";
                var folders = filePath.Split(System.IO.Path.DirectorySeparatorChar);

                var detections = image.detections;
                float max_confidence = (float)image.max_detection_conf;

                int currenteMammalCategory = eMammalCategory.blank;

                logImages.Append(imageName + "\n");
                logCount++;
                LogProcessedImages(ref logImages, ref logCount);

                if (folders.Length > 1)
                    imageplusLastFolderName = folders[folders.Length - 2].ToString() + "_" + imageName;

                int imageSequenceId = FindSequenceId(dtImageSequences, imageName, imageplusLastFolderName);
                progressCount++;

                // if the image is not in the eMammal database continue to next image
                if (imageSequenceId == -1)
                {
                    Common.ShowProgress(window, string.Format("image: {0} not found in deployment {1}",
                                        imageName, deploymentName), progressCount);
                    continue;
                }
                else
                {
                    foundImage = true;
                    if (imageNotFoundProgressSet == true)
                        Common.ShowProgress(window, Constants.PROGRESS_CONTINUING_WITH_NEXT_IMAGE,
                                            progressCount);
                }

                if (detections.Count() == 0)
                {
                    sql.AppendFormat("('{0}', '{1}', '{2}'),", imageSequenceId, currenteMammalCategory, 1);
                    count++;
                }

                if (progressCount % showProgressCount == 0)
                {
                    if (totalImages > imageSequenceCount)

                        Common.ShowProgress(window, string.Format("Processed {0} images",
                                            progressCount.ToString(), totalImages.ToString()), progressCount);
                    else
                        Common.ShowProgress(window, string.Format("Processed {0} out of {1} images",
                                            progressCount.ToString(), totalImages.ToString()), progressCount);

                }

                EnumerateDetections(eMammalCategory, ref count, ref sql, detections, max_confidence, ref currenteMammalCategory, imageSequenceId);

                if (count >= maxBulkInsertCount)
                {
                    logger.Info("Inserting {0} detections", maxBulkInsertCount.ToString());
                    count = 0;

                    bool success = db.BulkInsertAnnotations(sql);
                    if (!success)
                        return false;

                    sql = db.GetBulkInsertInitialString();
                    recordsAdded = true;

                    Common.ShowProgress(window,
                        string.Format("Inserting {0} detections", maxBulkInsertCount.ToString()),
                        progressCount, false);
                }
            }
            if (logCount > 0)
                logger.Info(logImages.ToString());

            // Add remaining detections
            if (!recordsAdded & foundImage)
            {
                Common.ShowProgress(window, Constants.PROGRESS_UPDATING_ANNOTATIONS_IN_DB, progressCount);
                db.BulkInsertAnnotations(sql);

                progressCount++;

                if (data.images.Count < maxBulkInsertCount)
                {
                    logger.Info(Constants.INSERTING_DETECTIONS);
                    Common.ShowProgress(window, Constants.INSERTING_DETECTIONS, progressCount);
                }
                else
                {
                    logger.Info(Constants.INSERTING_REMAINING_DETECTIONS);
                    Common.ShowProgress(window, Constants.INSERTING_REMAINING_DETECTIONS, progressCount);
                }
            }

            // The deployment does not contain any images that is within the provided JSON file
            if (!foundImage)
            {
                logger.Info("No matching images found in " + deploymentName + " that match the image names in the provided JSON file");

                Common.SetMessage(window, "No matching images found in " + deploymentName + " that match the image names in the provided JSON file", true);
                return false;
            }

            logger.Info(Constants.ANNOTATIONS_ADDED_FOR_ALL_IMAGES);

            //ShowProgress((int)window.ProgressbarUpdateProgress.Maximum, Constants.ANNOTATIONS_ADDED_FOR_ALL_IMAGES, true, true);
            Common.ShowProgress(window, Constants.ANNOTATIONS_ADDED_FOR_ALL_IMAGES, (int)window.ProgressbarUpdateProgress.Maximum);
            Common.delay();

            db.CloseConnection();

            return true;
        }

        private void LogProcessedImages(ref StringBuilder logImages, ref int logCount)
        {
            if (logCount > 100)
            {
                logger.Info(logImages.ToString());
                logImages = new StringBuilder();
                logCount = 0;
            }
        }

       
        /// <summary>
        /// Enumerate detections and udpate sql query
        /// </summary>
        /// <param name="eMammalCategory"></param>
        /// <param name="count"></param>
        /// <param name="sql"></param>
        /// <param name="detections"></param>
        /// <param name="max_confidence"></param>
        /// <param name="currenteMammalCategory"></param>
        /// <param name="imageSequenceId"></param>
        private static void EnumerateDetections(Category eMammalCategory, ref int count, ref StringBuilder sql,
            Detection[] detections, float max_confidence, ref int currenteMammalCategory, int imageSequenceId)
        {
            foreach (var d in detections)
            {
                // TODO: confirm json file is reading in detections correctly
                if ((float)d.conf != max_confidence)
                    continue;

                // map to selected eMammal categories
                if (d.category == animal)
                    currenteMammalCategory = eMammalCategory.animal;

                else if (d.category == person)
                    currenteMammalCategory = eMammalCategory.person;

                else if (d.category == vehicle)
                    currenteMammalCategory = eMammalCategory.vehicle;

                sql.AppendFormat("('{0}', '{1}', '{2}'),", imageSequenceId, currenteMammalCategory, 1);

                count++;

            }
        }

        private int FindSequenceId(DataTable imageSequences, string imageName, string lastFolderName)
        {
            foreach (DataRow row in imageSequences.Rows)
            {
                if (row["raw_name"].ToString() == imageName)
                    return (int)row["image_sequence_id"];

                if (row["raw_name"].ToString() == lastFolderName)
                    return (int)row["image_sequence_id"];
            }
            return -1;
        }
        public bool VerifyAnnotations(int deploymentId)
        {
            DataTable dt = db.GetImagesForDeployment(deploymentId);

            StringBuilder logInfo = new StringBuilder();

            int count = 0;
            int totalImages = dt.Rows.Count;
            window.ProgressbarUpdateProgress.Maximum = dt.Rows.Count;
            int progressCount = 0;
            int showProgressCount = 10;

            window.TextBlockInfo.Visibility = Visibility.Hidden;

            showProgressCount = Common.GetShowProgressCount(showProgressCount, totalImages);

            foreach (DataRow row in dt.Rows)
            {
                progressCount++;
                string annotation = row[0].ToString() + " - " + row[3].ToString();

                window.RichTextBoxResults.AppendText(annotation + "\n");

                logInfo.Append(annotation);

                count++;

                if (count > showProgressCount)
                {
                    logger.Info(logInfo.ToString());
                    logInfo = new StringBuilder();
                    count = 0;

                    Common.ShowProgress(window,
                                        string.Format("Enumerating {0} annotations out of {1}", 
                                        progressCount.ToString(), 
                                        totalImages.ToString()),
                                        progressCount);
                }
            }

            if (logInfo.Length > 0)
            {
                logger.Info(logInfo.ToString());
            }
            return true;

        }
    }
}
