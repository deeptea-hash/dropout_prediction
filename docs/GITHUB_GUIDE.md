# Pushing this project to GitHub

## One-time prerequisites
1. Install Git: https://git-scm.com/downloads
2. Create a free account: https://github.com
3. Tell Git who you are (once per computer):
   ```bash
   git config --global user.name  "Your Name"
   git config --global user.email "you@example.com"
   ```

## Option A - VS Code (no commands)
1. Open the `dropout-prediction` folder in VS Code.
2. Click the **Source Control** icon (left bar) -> **Initialize Repository**.
3. Type a message such as `Initial commit: dropout prediction project`, click **Commit** (stage all if asked).
4. Click **Publish Branch** -> sign in to GitHub -> choose **Public** or **Private** -> done.

## Option B - terminal
1. On github.com click **New repository**, name it `dropout-prediction`, leave it EMPTY
   (no README / .gitignore / licence), click **Create repository**.
2. In the VS Code terminal, inside the project folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: dropout prediction project"
   git branch -M main
   git remote add origin https://github.com/<your-username>/dropout-prediction.git
   git push -u origin main
   ```
   If asked to log in, use your GitHub account (a browser window or a Personal Access Token will be requested).

## Later changes
```bash
git add .
git commit -m "Describe what you changed"
git push
```

## Check before pushing
- `data/` contains no private or licensed data (the `.gitignore` already excludes CSV files).
- `outputs/` is excluded; commit selected figures only if you want them on GitHub (copy to `docs/`).
- If three of you contribute: add teammates under **Settings -> Collaborators**, and each person commits from their own account.
- After the first push, the **Actions** tab shows the automated tests (`.github/workflows/tests.yml`) running.
