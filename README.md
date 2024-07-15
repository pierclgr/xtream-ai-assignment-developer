# xtream AI Challenge - Software Engineer

## Ready Player 1? 🚀

Hey there! Congrats on crushing our first screening! 🎉 You're off to a fantastic start!

Welcome to the next level of your journey to join the [xtream](https://xtreamers.io) AI squad. Here's your next mission.

You will face 4 challenges. **Don't stress about doing them all**. Just dive into the ones that spark your interest or that you feel confident about. Let your talents shine bright! ✨

This assignment is designed to test your skills in engineering and software development. You **will not need to design or develop models**. Someone has already done that for you. 

You've got **7 days** to show us your magic, starting now. No rush—work at your own pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### Your Mission
[comment]: # (Well, well, well. Nice to see you around! You found an Easter Egg! Put the picture of an iguana at the beginning of the "How to Run" section, just to let us know. And have fun with the challenges! 🦎)

Think of this as a real-world project. Fork this repo and treat it like you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

**Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the problem and craft your own effective solutions.

---

### Context

Marta, a data scientist at xtream, has been working on a project for a client. She's been doing a great job, but she's got a lot on her plate. So, she's asked you to help her out with this project.

Marta has given you a notebook with the work she's done so far and a dataset to work with. You can find both in this repository.
You can also find a copy of the notebook on Google Colab [here](https://colab.research.google.com/drive/1ZUg5sAj-nW0k3E5fEcDuDBdQF-IhTQrd?usp=sharing).

The model is good enough; now it's time to build the supporting infrastructure.

### Challenge 1

**Develop an automated pipeline** that trains your model with fresh data, keeping it as sharp as the diamonds it processes. 
Pick the best linear model: do not worry about the xgboost model or hyperparameter tuning. 
Maintain a history of all the models you train and save the performance metrics of each one.

### Challenge 2

Level up! Now you need to support **both models** that Marta has developed: the linear regression and the XGBoost with hyperparameter optimization. 
Be careful. 
In the near future, you may want to include more models, so make sure your pipeline is flexible enough to handle that.

### Challenge 3

Build a **REST API** to integrate your model into a web app, making it a breeze for the team to use. Keep it developer-friendly – not everyone speaks 'data scientist'! 
Your API should support two use cases:
1. Predict the value of a diamond.
2. Given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight.

### Challenge 4

Observability is key. Save every request and response made to the APIs to a **proper database**.

---

## How to run
In order to run the application, it is necessary to create a Python virtual environment with the given requirements file. It is enough to open a terminal, move to the repository directory and run the following commands

```bash
python -m venv venv

# run if on Linux/unix
source venv/bin/activate

# run if on windows
venv\Scripts\activate

pip install -r requirements.txt
```

Then, in order to run the training it is enough to execute the script *main.py* with the following command
```bash
# run if on Linux/unix
source venv/bin/activate

# run if on windows
venv\Scripts\activate

python main.py
```

If you want to perform some other training tests, it is enough to edit the configuration file in *config/training_pipeline.json* with the parameters you need for the training. Then, by running the script *main.py*, it will perform a training for each *.csv* in the folder specified in the configuration.

Finally, to run the app, it is enough to execute the following commands
```bash
# run if on Linux/unix
source venv/bin/activate

# run if on windows
venv\Scripts\activate

python app.py
```

The app automatically selects the best available model among the trained ones.
Then, by heading to [http:0.0.0.0:8000/docs](http:0.0.0.0:8000/docs), you will access Swagger UI to test and query the app API.

You can predict a value for a diamond by testing the /predict endpoint with a request similar to
```json
{
    "carat": 1.0,
    "cut": "Ideal",
    "color": "E",
    "clarity": "VVS1",
    "depth": 61.0,
    "table": 55.0,
    "x": 6.5,
    "y": 6.5,
    "z": 4.0
}
```

You can also, given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight, with a request to the /samples endpoint similar to
```json
{
    "cut": "Ideal",
    "color": "E",
    "clarity": "VVS1",
    "weight": 1.0,
    "n_samples": 5
}
```

Finally, by accessing the /logs endpoint, you can get the list of sent requests.
 
