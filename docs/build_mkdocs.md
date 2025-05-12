# Building the MkDocs Site

To build the MkDocs site locally, follow these steps:

1. **Install MkDocs and Dependencies**:
   Ensure you have Python installed. Then, install MkDocs and the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Build the Site**:
   Run the following command to build the site:
   ```bash
   mkdocs build
   ```
   This will generate the static site in the `site/` directory.

3. **Serve the Site Locally**:
   To preview the site locally, use:
   ```bash
   mkdocs serve
   ```
   This will start a local development server, and you can view the site at `http://127.0.0.1:8000/`.

4. **Custom Documentation Directory**:
   The site's page files are located in the `mkdocs/` directory. This is specified in the `mkdocs.yml` file under the `docs_dir` key:
   ```yaml
   docs_dir: mkdocs
   ```
   This is so it doesn't clash with Pytorch Wildlife's preexisting `docs/` directory (so no breaking changes!)
   
   Ensure that any changes to the documentation are made in the `mkdocs/` directory.

5. **Exclude the `site/` Directory**:
   The `site/` directory is automatically generated and should not be included in version control. It is already added to `.gitignore`.

