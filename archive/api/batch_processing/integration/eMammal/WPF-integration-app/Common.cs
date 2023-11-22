using System;
using System.Drawing;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;

namespace eMammal_integration_application
{
    public class Common
    {
        public static void CheckConnection(eMammalIntegrationWindow window, bool loadProject = false)
        {
            eMammalMySQLOps db = new eMammalMySQLOps();

            bool isConnectionOpen = false;
            while (isConnectionOpen == false)
            {
                Thread.Sleep(200);
                isConnectionOpen = db.OpenConnectionIfNotOpen(true);
            }
            window.Dispatcher.BeginInvoke(new Action(() =>
            {
                Common.SetMessage(window, Constants.DATABASE_AVAILABLE, false, false);

                window.Tab.Visibility = Visibility.Visible;

                if (loadProject)
                    window.Loadproject();

                window.Tab.SelectedIndex = 0;
                window.Tab.IsEnabled = true;

                window.IsEnabled = true;

                window.ButtonBack.Visibility = Visibility.Hidden;
                window.ReactivateButton(window.ButtonNext);
                window.ReactivateButton(window.ButtonBrowse);

            }));
        }
        public static void SetMessage(eMammalIntegrationWindow window, string msg, bool isError = false, bool showMessageBox = true)
        {
            //window.Visibility = Visibility.Visible;
            //window.TextBlockInfo.Text = msg;
            //TextBlock.Text = msg;

            window.TextBlockInfo.Dispatcher.Invoke(() => window.TextBlockInfo.Visibility = Visibility.Visible, DispatcherPriority.Background);
            window.TextBlockInfo.Dispatcher.Invoke(() => window.TextBlockInfo.Text = msg, DispatcherPriority.Normal);

            if (isError)
                window.Foreground = new SolidColorBrush(Colors.Red);
            else
                window.Foreground = new SolidColorBrush(Colors.Blue);


            if (showMessageBox)
                SetMessageBox(msg, isError);

        }

        public static void SetMessageBox(string msg, bool error = false)
        {
            //CustomMessageBox w = new CustomMessageBox();
            //w.LabelInfo.Content = msg;
            //w.ShowDialog();
            if (error)
                MessageBox.Show(msg, "", MessageBoxButton.OK,
                               MessageBoxImage.Error,
                               MessageBoxResult.OK,
                               MessageBoxOptions.DefaultDesktopOnly);
            else
                MessageBox.Show(msg, "", MessageBoxButton.OK,
                                MessageBoxImage.Information,
                                MessageBoxResult.OK,
                                MessageBoxOptions.DefaultDesktopOnly);
        }
        public static void ShowProgress(eMammalIntegrationWindow window, string msg, int progressCount,
            bool isLast = true, bool showProgressBar = true)
        {
            window.LabelProgress.Content = msg;

            window.LabelProgress.Dispatcher.Invoke(() =>
            window.LabelProgress.Visibility = Visibility.Visible, DispatcherPriority.Background);

            if (showProgressBar)
            {
                window.ProgressbarUpdateProgress.Dispatcher.Invoke(() =>
                window.ProgressbarUpdateProgress.Visibility = Visibility.Visible, DispatcherPriority.Background);
            }

            if (isLast)
                window.ProgressbarUpdateProgress.Dispatcher.Invoke(() =>
                window.ProgressbarUpdateProgress.Value = progressCount, DispatcherPriority.Normal);
            else
                window.ProgressbarUpdateProgress.Dispatcher.Invoke(() =>
                window.ProgressbarUpdateProgress.Value = progressCount, DispatcherPriority.Background);


        }
        public static void HideProgress(eMammalIntegrationWindow window)
        {
            window.LabelProgress.Content = "";
            window.LabelProgress.Visibility = Visibility.Hidden;
            window.ProgressbarUpdateProgress.Dispatcher.Invoke(() =>
            window.ProgressbarUpdateProgress.Visibility = Visibility.Hidden, DispatcherPriority.Normal);
            window.ProgressbarUpdateProgress.Dispatcher.Invoke(() =>
            window.ProgressbarUpdateProgress.Value = 0, DispatcherPriority.Background);
        }
        public static void delay(int maxCount = 1000000)
        {
            int count = 0;
            while (count < 1000000)
                count++;
        }
        public static int GetShowProgressCount(int showProgressCount, int totalImages)
        {
            if (totalImages < 10)
                showProgressCount = 1;

            else if (totalImages > 1000 && totalImages < 100000)
                showProgressCount = 100;

            else if (totalImages > 100000)
                showProgressCount = 1000;

            return showProgressCount;
        }
    }
}
