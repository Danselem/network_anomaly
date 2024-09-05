python3 -m venv .venv

Check the link for how to install cookiecutter: https://mlops-guide.github.io/Structure/starting/
cookiecutter https://dagshub.com/DagsHub/Cookiecutter-MLOps.git 


cp -r /Users/daniel/.cookiecutters/Cookiecutter-MLOps/ .

Link for git branch divergence: https://jvns.ca/blog/2024/02/01/dealing-with-diverged-git-branches/

```bash
git pull --rebase
git push --force
git reset --hard origin/main
```

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

https://www.warp.dev/terminus/undo-git-add

git push origin HEAD:main

Write a blog on Linux commands