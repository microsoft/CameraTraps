using System.Drawing;
using System.Windows.Forms;

namespace CameraTrapJsonManagerApp
{
    partial class Form
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form));
            this.openFileDialog = new System.Windows.Forms.OpenFileDialog();
            this.backgroundWorkerMain = new System.ComponentModel.BackgroundWorker();
            this.backgroundWorkerProgressReporter = new System.ComponentModel.BackgroundWorker();
            this.buttonBrowse = new System.Windows.Forms.Button();
            this.textboxConfidenceThreshold = new System.Windows.Forms.TextBox();
            this.lblInputFile = new System.Windows.Forms.Label();
            this.lblConfidenceThreshold = new System.Windows.Forms.Label();
            this.buttonSubmit = new System.Windows.Forms.Button();
            this.textboxQuery = new System.Windows.Forms.TextBox();
            this.labelQuery = new System.Windows.Forms.Label();
            this.textboxReplacement = new System.Windows.Forms.TextBox();
            this.labelReplacement = new System.Windows.Forms.Label();
            this.comboBoxSplitFolderMode = new System.Windows.Forms.ComboBox();
            this.labelSplitFolderMode = new System.Windows.Forms.Label();
            this.textboxOutputFolderFile = new System.Windows.Forms.TextBox();
            this.lblOutputFolderFileName = new System.Windows.Forms.Label();
            this.textboxInputFile = new System.Windows.Forms.TextBox();
            this.panelMain = new System.Windows.Forms.Panel();
            this.buttonBrowseFolder = new System.Windows.Forms.Button();
            this.buttonHelp = new System.Windows.Forms.Button();
            this.panel1 = new System.Windows.Forms.Panel();
            this.checkBoxCopyJsonstoFolders = new System.Windows.Forms.CheckBox();
            this.checkBoxCreateFolders = new System.Windows.Forms.CheckBox();
            this.checkBoxMakeFolderRelative = new System.Windows.Forms.CheckBox();
            this.checkBoxOverwriteJsonFiles = new System.Windows.Forms.CheckBox();
            this.labelConfidenceThresholdHelpInfo = new System.Windows.Forms.Label();
            this.panelSplitFolderMode = new System.Windows.Forms.Panel();
            this.textBoxSplitParameter = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.checkBoxSplitFolders = new System.Windows.Forms.CheckBox();
            this.statusGroupBox = new System.Windows.Forms.Panel();
            this.progressBar1 = new System.Windows.Forms.ProgressBar();
            this.labelProgressPercentage = new System.Windows.Forms.Label();
            this.richTextboxStatus = new System.Windows.Forms.RichTextBox();
            this.labelProgressMsg = new System.Windows.Forms.Label();
            this.folderBrowserDialog = new System.Windows.Forms.FolderBrowserDialog();
            this.panelMain.SuspendLayout();
            this.panel1.SuspendLayout();
            this.panelSplitFolderMode.SuspendLayout();
            this.statusGroupBox.SuspendLayout();
            this.SuspendLayout();
            // 
            // backgroundWorkerMain
            // 
            this.backgroundWorkerMain.WorkerReportsProgress = true;
            this.backgroundWorkerMain.DoWork += new System.ComponentModel.DoWorkEventHandler(this.backgroundWorker1_DoWork);
            this.backgroundWorkerMain.RunWorkerCompleted += new System.ComponentModel.RunWorkerCompletedEventHandler(this.backgroundWorkerMain_RunWorkerCompleted);
            // 
            // backgroundWorkerProgressReporter
            // 
            this.backgroundWorkerProgressReporter.WorkerReportsProgress = true;
            this.backgroundWorkerProgressReporter.ProgressChanged += new System.ComponentModel.ProgressChangedEventHandler(this.backgroundWorker2_ProgressChanged);
            // 
            // buttonBrowse
            // 
            this.buttonBrowse.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            this.buttonBrowse.Font = new System.Drawing.Font("Microsoft Sans Serif", 10.125F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonBrowse.ForeColor = System.Drawing.Color.White;
            this.buttonBrowse.Location = new System.Drawing.Point(820, 60);
            this.buttonBrowse.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.buttonBrowse.Name = "buttonBrowse";
            this.buttonBrowse.Size = new System.Drawing.Size(115, 42);
            this.buttonBrowse.TabIndex = 68;
            this.buttonBrowse.Text = "Browse";
            this.buttonBrowse.UseVisualStyleBackColor = false;
            this.buttonBrowse.Click += new System.EventHandler(this.buttonBrowse_Click);
            // 
            // textboxConfidenceThreshold
            // 
            this.textboxConfidenceThreshold.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textboxConfidenceThreshold.ForeColor = System.Drawing.Color.Black;
            this.textboxConfidenceThreshold.Location = new System.Drawing.Point(259, 315);
            this.textboxConfidenceThreshold.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.textboxConfidenceThreshold.Name = "textboxConfidenceThreshold";
            this.textboxConfidenceThreshold.Size = new System.Drawing.Size(103, 32);
            this.textboxConfidenceThreshold.TabIndex = 69;
            this.textboxConfidenceThreshold.TextChanged += new System.EventHandler(this.textboxConfidenceThreshold_TextChanged);
            // 
            // lblInputFile
            // 
            this.lblInputFile.AutoSize = true;
            this.lblInputFile.BackColor = System.Drawing.Color.Transparent;
            this.lblInputFile.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblInputFile.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.lblInputFile.Location = new System.Drawing.Point(5, 69);
            this.lblInputFile.Name = "lblInputFile";
            this.lblInputFile.Size = new System.Drawing.Size(92, 24);
            this.lblInputFile.TabIndex = 70;
            this.lblInputFile.Text = "Input file:";
            // 
            // lblConfidenceThreshold
            // 
            this.lblConfidenceThreshold.AutoSize = true;
            this.lblConfidenceThreshold.BackColor = System.Drawing.Color.Transparent;
            this.lblConfidenceThreshold.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblConfidenceThreshold.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.lblConfidenceThreshold.Location = new System.Drawing.Point(5, 320);
            this.lblConfidenceThreshold.Name = "lblConfidenceThreshold";
            this.lblConfidenceThreshold.Size = new System.Drawing.Size(196, 24);
            this.lblConfidenceThreshold.TabIndex = 71;
            this.lblConfidenceThreshold.Text = "Confidence threshold:";
            // 
            // buttonSubmit
            // 
            this.buttonSubmit.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            this.buttonSubmit.FlatAppearance.BorderColor = System.Drawing.Color.RoyalBlue;
            this.buttonSubmit.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonSubmit.ForeColor = System.Drawing.Color.White;
            this.buttonSubmit.Location = new System.Drawing.Point(7, 556);
            this.buttonSubmit.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.buttonSubmit.Name = "buttonSubmit";
            this.buttonSubmit.Size = new System.Drawing.Size(656, 55);
            this.buttonSubmit.TabIndex = 72;
            this.buttonSubmit.Text = "Process";
            this.buttonSubmit.UseVisualStyleBackColor = false;
            this.buttonSubmit.Click += new System.EventHandler(this.buttonSubmit_Click);
            // 
            // textboxQuery
            // 
            this.textboxQuery.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textboxQuery.ForeColor = System.Drawing.Color.Black;
            this.textboxQuery.Location = new System.Drawing.Point(259, 184);
            this.textboxQuery.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.textboxQuery.Name = "textboxQuery";
            this.textboxQuery.Size = new System.Drawing.Size(257, 32);
            this.textboxQuery.TabIndex = 73;
            // 
            // labelQuery
            // 
            this.labelQuery.AutoSize = true;
            this.labelQuery.BackColor = System.Drawing.Color.Transparent;
            this.labelQuery.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelQuery.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.labelQuery.Location = new System.Drawing.Point(5, 192);
            this.labelQuery.Name = "labelQuery";
            this.labelQuery.Size = new System.Drawing.Size(67, 24);
            this.labelQuery.TabIndex = 74;
            this.labelQuery.Text = "Query:";
            // 
            // textboxReplacement
            // 
            this.textboxReplacement.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textboxReplacement.ForeColor = System.Drawing.Color.Black;
            this.textboxReplacement.Location = new System.Drawing.Point(259, 236);
            this.textboxReplacement.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.textboxReplacement.Name = "textboxReplacement";
            this.textboxReplacement.Size = new System.Drawing.Size(257, 32);
            this.textboxReplacement.TabIndex = 75;
            // 
            // labelReplacement
            // 
            this.labelReplacement.AutoSize = true;
            this.labelReplacement.BackColor = System.Drawing.Color.Transparent;
            this.labelReplacement.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelReplacement.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.labelReplacement.Location = new System.Drawing.Point(5, 236);
            this.labelReplacement.Name = "labelReplacement";
            this.labelReplacement.Size = new System.Drawing.Size(125, 24);
            this.labelReplacement.TabIndex = 76;
            this.labelReplacement.Text = "Replacement:";
            // 
            // comboBoxSplitFolderMode
            // 
            this.comboBoxSplitFolderMode.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxSplitFolderMode.FlatStyle = System.Windows.Forms.FlatStyle.Popup;
            this.comboBoxSplitFolderMode.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.comboBoxSplitFolderMode.ForeColor = System.Drawing.Color.Black;
            this.comboBoxSplitFolderMode.FormattingEnabled = true;
            this.comboBoxSplitFolderMode.Items.AddRange(new object[] {
            "Top",
            "Bottom",
            "NFromBottom",
            "NFromTop"});
            this.comboBoxSplitFolderMode.Location = new System.Drawing.Point(373, 15);
            this.comboBoxSplitFolderMode.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.comboBoxSplitFolderMode.Name = "comboBoxSplitFolderMode";
            this.comboBoxSplitFolderMode.Size = new System.Drawing.Size(175, 32);
            this.comboBoxSplitFolderMode.TabIndex = 77;
            // 
            // labelSplitFolderMode
            // 
            this.labelSplitFolderMode.AutoSize = true;
            this.labelSplitFolderMode.BackColor = System.Drawing.Color.Transparent;
            this.labelSplitFolderMode.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelSplitFolderMode.ForeColor = System.Drawing.Color.Black;
            this.labelSplitFolderMode.Location = new System.Drawing.Point(176, 18);
            this.labelSplitFolderMode.Name = "labelSplitFolderMode";
            this.labelSplitFolderMode.Size = new System.Drawing.Size(160, 24);
            this.labelSplitFolderMode.TabIndex = 78;
            this.labelSplitFolderMode.Text = "Split folder mode:";
            // 
            // textboxOutputFolderFile
            // 
            this.textboxOutputFolderFile.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textboxOutputFolderFile.ForeColor = System.Drawing.Color.Black;
            this.textboxOutputFolderFile.Location = new System.Drawing.Point(259, 129);
            this.textboxOutputFolderFile.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.textboxOutputFolderFile.Name = "textboxOutputFolderFile";
            this.textboxOutputFolderFile.Size = new System.Drawing.Size(543, 32);
            this.textboxOutputFolderFile.TabIndex = 83;
            this.textboxOutputFolderFile.TextChanged += new System.EventHandler(this.textboxOutputFolderFile_TextChanged);
            // 
            // lblOutputFolderFileName
            // 
            this.lblOutputFolderFileName.AutoSize = true;
            this.lblOutputFolderFileName.BackColor = System.Drawing.Color.Transparent;
            this.lblOutputFolderFileName.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblOutputFolderFileName.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.lblOutputFolderFileName.Location = new System.Drawing.Point(5, 114);
            this.lblOutputFolderFileName.Name = "lblOutputFolderFileName";
            this.lblOutputFolderFileName.Size = new System.Drawing.Size(224, 48);
            this.lblOutputFolderFileName.TabIndex = 84;
            this.lblOutputFolderFileName.Text = "Enter output file / \nbrowse to select a folder:";
            // 
            // textboxInputFile
            // 
            this.textboxInputFile.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textboxInputFile.ForeColor = System.Drawing.Color.Black;
            this.textboxInputFile.Location = new System.Drawing.Point(259, 65);
            this.textboxInputFile.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.textboxInputFile.Name = "textboxInputFile";
            this.textboxInputFile.Size = new System.Drawing.Size(543, 32);
            this.textboxInputFile.TabIndex = 85;
            this.textboxInputFile.TextChanged += new System.EventHandler(this.textboxInputFile_TextChanged);
            // 
            // panelMain
            // 
            this.panelMain.BackColor = System.Drawing.Color.Transparent;
            this.panelMain.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.panelMain.Controls.Add(this.buttonBrowseFolder);
            this.panelMain.Controls.Add(this.buttonHelp);
            this.panelMain.Controls.Add(this.panel1);
            this.panelMain.Controls.Add(this.labelConfidenceThresholdHelpInfo);
            this.panelMain.Controls.Add(this.panelSplitFolderMode);
            this.panelMain.Controls.Add(this.textboxInputFile);
            this.panelMain.Controls.Add(this.lblOutputFolderFileName);
            this.panelMain.Controls.Add(this.textboxOutputFolderFile);
            this.panelMain.Controls.Add(this.labelReplacement);
            this.panelMain.Controls.Add(this.textboxReplacement);
            this.panelMain.Controls.Add(this.labelQuery);
            this.panelMain.Controls.Add(this.textboxQuery);
            this.panelMain.Controls.Add(this.buttonSubmit);
            this.panelMain.Controls.Add(this.lblConfidenceThreshold);
            this.panelMain.Controls.Add(this.lblInputFile);
            this.panelMain.Controls.Add(this.textboxConfidenceThreshold);
            this.panelMain.Controls.Add(this.buttonBrowse);
            this.panelMain.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.panelMain.Location = new System.Drawing.Point(40, 17);
            this.panelMain.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.panelMain.Name = "panelMain";
            this.panelMain.Size = new System.Drawing.Size(1033, 703);
            this.panelMain.TabIndex = 68;
            // 
            // buttonBrowseFolder
            // 
            this.buttonBrowseFolder.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            this.buttonBrowseFolder.Font = new System.Drawing.Font("Microsoft Sans Serif", 10.125F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonBrowseFolder.ForeColor = System.Drawing.Color.White;
            this.buttonBrowseFolder.Location = new System.Drawing.Point(819, 124);
            this.buttonBrowseFolder.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.buttonBrowseFolder.Name = "buttonBrowseFolder";
            this.buttonBrowseFolder.Size = new System.Drawing.Size(115, 42);
            this.buttonBrowseFolder.TabIndex = 99;
            this.buttonBrowseFolder.Text = "Browse";
            this.buttonBrowseFolder.UseVisualStyleBackColor = false;
            this.buttonBrowseFolder.Click += new System.EventHandler(this.buttonBrowseFolder_Click);
            // 
            // buttonHelp
            // 
            this.buttonHelp.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            this.buttonHelp.FlatAppearance.BorderColor = System.Drawing.Color.RoyalBlue;
            this.buttonHelp.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonHelp.ForeColor = System.Drawing.Color.White;
            this.buttonHelp.Location = new System.Drawing.Point(668, 556);
            this.buttonHelp.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.buttonHelp.Name = "buttonHelp";
            this.buttonHelp.Size = new System.Drawing.Size(225, 55);
            this.buttonHelp.TabIndex = 98;
            this.buttonHelp.Text = "Help";
            this.buttonHelp.UseVisualStyleBackColor = false;
            this.buttonHelp.Click += new System.EventHandler(this.ButtonHelp_Click);
            // 
            // panel1
            // 
            this.panel1.BackColor = System.Drawing.Color.LightSteelBlue;
            this.panel1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel1.Controls.Add(this.checkBoxCopyJsonstoFolders);
            this.panel1.Controls.Add(this.checkBoxCreateFolders);
            this.panel1.Controls.Add(this.checkBoxMakeFolderRelative);
            this.panel1.Controls.Add(this.checkBoxOverwriteJsonFiles);
            this.panel1.Location = new System.Drawing.Point(5, 468);
            this.panel1.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(882, 65);
            this.panel1.TabIndex = 97;
            // 
            // checkBoxCopyJsonstoFolders
            // 
            this.checkBoxCopyJsonstoFolders.AutoSize = true;
            this.checkBoxCopyJsonstoFolders.BackColor = System.Drawing.Color.Transparent;
            this.checkBoxCopyJsonstoFolders.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxCopyJsonstoFolders.ForeColor = System.Drawing.Color.Black;
            this.checkBoxCopyJsonstoFolders.Location = new System.Drawing.Point(0, 4);
            this.checkBoxCopyJsonstoFolders.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.checkBoxCopyJsonstoFolders.Name = "checkBoxCopyJsonstoFolders";
            this.checkBoxCopyJsonstoFolders.Padding = new System.Windows.Forms.Padding(13, 14, 13, 14);
            this.checkBoxCopyJsonstoFolders.Size = new System.Drawing.Size(234, 56);
            this.checkBoxCopyJsonstoFolders.TabIndex = 92;
            this.checkBoxCopyJsonstoFolders.Text = "Copy jsons to folders";
            this.checkBoxCopyJsonstoFolders.UseVisualStyleBackColor = false;
            this.checkBoxCopyJsonstoFolders.CheckedChanged += new System.EventHandler(this.checkBoxCopyJsonstoFolders_CheckedChanged);
            // 
            // checkBoxCreateFolders
            // 
            this.checkBoxCreateFolders.AutoSize = true;
            this.checkBoxCreateFolders.BackColor = System.Drawing.Color.Transparent;
            this.checkBoxCreateFolders.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxCreateFolders.ForeColor = System.Drawing.Color.Black;
            this.checkBoxCreateFolders.Location = new System.Drawing.Point(237, 4);
            this.checkBoxCreateFolders.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.checkBoxCreateFolders.Name = "checkBoxCreateFolders";
            this.checkBoxCreateFolders.Padding = new System.Windows.Forms.Padding(13, 14, 8, 7);
            this.checkBoxCreateFolders.Size = new System.Drawing.Size(171, 49);
            this.checkBoxCreateFolders.TabIndex = 93;
            this.checkBoxCreateFolders.Text = "Create folders";
            this.checkBoxCreateFolders.UseVisualStyleBackColor = false;
            // 
            // checkBoxMakeFolderRelative
            // 
            this.checkBoxMakeFolderRelative.AutoSize = true;
            this.checkBoxMakeFolderRelative.BackColor = System.Drawing.Color.Transparent;
            this.checkBoxMakeFolderRelative.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxMakeFolderRelative.ForeColor = System.Drawing.Color.Black;
            this.checkBoxMakeFolderRelative.Location = new System.Drawing.Point(653, 4);
            this.checkBoxMakeFolderRelative.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.checkBoxMakeFolderRelative.Name = "checkBoxMakeFolderRelative";
            this.checkBoxMakeFolderRelative.Padding = new System.Windows.Forms.Padding(13, 14, 8, 7);
            this.checkBoxMakeFolderRelative.Size = new System.Drawing.Size(220, 49);
            this.checkBoxMakeFolderRelative.TabIndex = 90;
            this.checkBoxMakeFolderRelative.Text = "Make folder-relative";
            this.checkBoxMakeFolderRelative.UseVisualStyleBackColor = false;
            // 
            // checkBoxOverwriteJsonFiles
            // 
            this.checkBoxOverwriteJsonFiles.AutoSize = true;
            this.checkBoxOverwriteJsonFiles.BackColor = System.Drawing.Color.Transparent;
            this.checkBoxOverwriteJsonFiles.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxOverwriteJsonFiles.ForeColor = System.Drawing.Color.Black;
            this.checkBoxOverwriteJsonFiles.Location = new System.Drawing.Point(421, 4);
            this.checkBoxOverwriteJsonFiles.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.checkBoxOverwriteJsonFiles.Name = "checkBoxOverwriteJsonFiles";
            this.checkBoxOverwriteJsonFiles.Padding = new System.Windows.Forms.Padding(13, 14, 8, 7);
            this.checkBoxOverwriteJsonFiles.Size = new System.Drawing.Size(214, 49);
            this.checkBoxOverwriteJsonFiles.TabIndex = 91;
            this.checkBoxOverwriteJsonFiles.Text = "Overwrite json files";
            this.checkBoxOverwriteJsonFiles.UseVisualStyleBackColor = false;
            // 
            // labelConfidenceThresholdHelpInfo
            // 
            this.labelConfidenceThresholdHelpInfo.AutoSize = true;
            this.labelConfidenceThresholdHelpInfo.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelConfidenceThresholdHelpInfo.ForeColor = System.Drawing.Color.WhiteSmoke;
            this.labelConfidenceThresholdHelpInfo.Location = new System.Drawing.Point(370, 318);
            this.labelConfidenceThresholdHelpInfo.Name = "labelConfidenceThresholdHelpInfo";
            this.labelConfidenceThresholdHelpInfo.Size = new System.Drawing.Size(213, 24);
            this.labelConfidenceThresholdHelpInfo.TabIndex = 96;
            this.labelConfidenceThresholdHelpInfo.Text = "(value between 0 and 1)";
            // 
            // panelSplitFolderMode
            // 
            this.panelSplitFolderMode.BackColor = System.Drawing.Color.LightSteelBlue;
            this.panelSplitFolderMode.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelSplitFolderMode.Controls.Add(this.textBoxSplitParameter);
            this.panelSplitFolderMode.Controls.Add(this.label1);
            this.panelSplitFolderMode.Controls.Add(this.labelSplitFolderMode);
            this.panelSplitFolderMode.Controls.Add(this.checkBoxSplitFolders);
            this.panelSplitFolderMode.Controls.Add(this.comboBoxSplitFolderMode);
            this.panelSplitFolderMode.ForeColor = System.Drawing.Color.Black;
            this.panelSplitFolderMode.Location = new System.Drawing.Point(5, 389);
            this.panelSplitFolderMode.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.panelSplitFolderMode.Name = "panelSplitFolderMode";
            this.panelSplitFolderMode.Size = new System.Drawing.Size(882, 65);
            this.panelSplitFolderMode.TabIndex = 95;
            this.panelSplitFolderMode.Paint += new System.Windows.Forms.PaintEventHandler(this.panelSplitFolderMode_Paint);
            // 
            // textBoxSplitParameter
            // 
            this.textBoxSplitParameter.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxSplitParameter.ForeColor = System.Drawing.Color.Black;
            this.textBoxSplitParameter.Location = new System.Drawing.Point(748, 15);
            this.textBoxSplitParameter.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.textBoxSplitParameter.Name = "textBoxSplitParameter";
            this.textBoxSplitParameter.Size = new System.Drawing.Size(103, 32);
            this.textBoxSplitParameter.TabIndex = 91;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.BackColor = System.Drawing.Color.Transparent;
            this.label1.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.ForeColor = System.Drawing.Color.Black;
            this.label1.Location = new System.Drawing.Point(581, 18);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(145, 24);
            this.label1.TabIndex = 90;
            this.label1.Text = "Split parameter:";
            // 
            // checkBoxSplitFolders
            // 
            this.checkBoxSplitFolders.AutoSize = true;
            this.checkBoxSplitFolders.BackColor = System.Drawing.Color.Transparent;
            this.checkBoxSplitFolders.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxSplitFolders.ForeColor = System.Drawing.Color.Black;
            this.checkBoxSplitFolders.Location = new System.Drawing.Point(3, 4);
            this.checkBoxSplitFolders.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.checkBoxSplitFolders.Name = "checkBoxSplitFolders";
            this.checkBoxSplitFolders.Padding = new System.Windows.Forms.Padding(13, 14, 8, 7);
            this.checkBoxSplitFolders.Size = new System.Drawing.Size(153, 49);
            this.checkBoxSplitFolders.TabIndex = 89;
            this.checkBoxSplitFolders.Text = "Split folders";
            this.checkBoxSplitFolders.UseVisualStyleBackColor = false;
            // 
            // statusGroupBox
            // 
            this.statusGroupBox.BackColor = System.Drawing.Color.White;
            this.statusGroupBox.Controls.Add(this.progressBar1);
            this.statusGroupBox.Controls.Add(this.labelProgressPercentage);
            this.statusGroupBox.Controls.Add(this.richTextboxStatus);
            this.statusGroupBox.Controls.Add(this.labelProgressMsg);
            this.statusGroupBox.Font = new System.Drawing.Font("Microsoft Sans Serif", 11F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.statusGroupBox.Location = new System.Drawing.Point(1, 658);
            this.statusGroupBox.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.statusGroupBox.Name = "statusGroupBox";
            this.statusGroupBox.Padding = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.statusGroupBox.Size = new System.Drawing.Size(1123, 286);
            this.statusGroupBox.TabIndex = 73;
            // 
            // progressBar1
            // 
            this.progressBar1.Location = new System.Drawing.Point(3, 48);
            this.progressBar1.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.progressBar1.Name = "progressBar1";
            this.progressBar1.Size = new System.Drawing.Size(1120, 28);
            this.progressBar1.TabIndex = 73;
            // 
            // labelProgressPercentage
            // 
            this.labelProgressPercentage.AutoSize = true;
            this.labelProgressPercentage.ForeColor = System.Drawing.Color.Black;
            this.labelProgressPercentage.Location = new System.Drawing.Point(4, 15);
            this.labelProgressPercentage.Name = "labelProgressPercentage";
            this.labelProgressPercentage.Size = new System.Drawing.Size(232, 24);
            this.labelProgressPercentage.TabIndex = 72;
            this.labelProgressPercentage.Text = "[labelProgressPercentage]";
            // 
            // richTextboxStatus
            // 
            this.richTextboxStatus.BackColor = System.Drawing.Color.Black;
            this.richTextboxStatus.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.richTextboxStatus.ForeColor = System.Drawing.Color.White;
            this.richTextboxStatus.Location = new System.Drawing.Point(3, 79);
            this.richTextboxStatus.Margin = new System.Windows.Forms.Padding(0);
            this.richTextboxStatus.Name = "richTextboxStatus";
            this.richTextboxStatus.ReadOnly = true;
            this.richTextboxStatus.Size = new System.Drawing.Size(1120, 219);
            this.richTextboxStatus.TabIndex = 71;
            this.richTextboxStatus.Text = "";
            // 
            // labelProgressMsg
            // 
            this.labelProgressMsg.AutoSize = true;
            this.labelProgressMsg.Font = new System.Drawing.Font("Calibri", 13.875F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelProgressMsg.ForeColor = System.Drawing.Color.MediumBlue;
            this.labelProgressMsg.Location = new System.Drawing.Point(5, 10);
            this.labelProgressMsg.Margin = new System.Windows.Forms.Padding(0);
            this.labelProgressMsg.Name = "labelProgressMsg";
            this.labelProgressMsg.Size = new System.Drawing.Size(199, 29);
            this.labelProgressMsg.TabIndex = 69;
            this.labelProgressMsg.Text = "[labelProgressMsg]";
            // 
            // Form
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.AutoSize = true;
            this.BackColor = System.Drawing.Color.White;
            this.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("$this.BackgroundImage")));
            this.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.ClientSize = new System.Drawing.Size(1120, 943);
            this.Controls.Add(this.statusGroupBox);
            this.Controls.Add(this.panelMain);
            this.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.MaximizeBox = false;
            this.Name = "Form";
            this.StartPosition = System.Windows.Forms.FormStartPosition.Manual;
            this.Text = "Camera Trap API Output Manager";
            this.Load += new System.EventHandler(this.form_Load);
            this.panelMain.ResumeLayout(false);
            this.panelMain.PerformLayout();
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.panelSplitFolderMode.ResumeLayout(false);
            this.panelSplitFolderMode.PerformLayout();
            this.statusGroupBox.ResumeLayout(false);
            this.statusGroupBox.PerformLayout();
            this.ResumeLayout(false);
        }

        #endregion
        private System.Windows.Forms.OpenFileDialog openFileDialog;
        private System.ComponentModel.BackgroundWorker backgroundWorkerMain;
        private System.ComponentModel.BackgroundWorker backgroundWorkerProgressReporter;
        private Button buttonBrowse;
        private TextBox textboxConfidenceThreshold;
        private Label lblInputFile;
        private Label lblConfidenceThreshold;
        private Button buttonSubmit;
        private TextBox textboxQuery;
        private Label labelQuery;
        private TextBox textboxReplacement;
        private Label labelReplacement;
        private ComboBox comboBoxSplitFolderMode;
        private Label labelSplitFolderMode;
        private TextBox textboxOutputFolderFile;
        private Label lblOutputFolderFileName;
        private TextBox textboxInputFile;
        private Panel panelMain;
        private CheckBox checkBoxSplitFolders;
        private CheckBox checkBoxCopyJsonstoFolders;
        private CheckBox checkBoxCreateFolders;
        private CheckBox checkBoxOverwriteJsonFiles;
        private CheckBox checkBoxMakeFolderRelative;
        private Panel panelSplitFolderMode;
        private Label labelConfidenceThresholdHelpInfo;
        private Panel panel1;
        private Panel statusGroupBox;
        private Label labelProgressPercentage;
        private RichTextBox richTextboxStatus;
        private Label labelProgressMsg;
        private ProgressBar progressBar1;
        private TextBox textBoxSplitParameter;
        private Label label1;
        private Button buttonHelp;
        private FolderBrowserDialog folderBrowserDialog;
        private Button buttonBrowseFolder;
    }
}