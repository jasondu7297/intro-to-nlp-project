# CS498: Introduction to NLP - Course Project

## Motivation: The Interstellar Autocomplete Challenge

Communications with astronauts in deep space missions (e.g., Mars) are constrained by severe bandwidth limitations and significant latency. Furthermore, inputting text on spacecraft control interfaces or suit-mounted displays can be cumbersome and time-consuming. 

Imagine an astronaut trying to log a critical observation or send a status update while wearing gloves, under stress, and with a communication window of only a few minutes. Every keystroke saved is precious time and bandwidth gained. 

Your mission is to build an intelligent auto-completion system—an advanced Language Model (LLM)-that can predict the astronaut's intended text as they type. By accurately predicting the next characters or words, your system will drastically reduce the effort required for communication and ensure that vital information reaches Earth efficiently.

You will be designing this system for an **unknown corpus**. We do not know exactly what the astronauts will be saying (though we know it involves communications between astronauts and mission control), and your model must be robust enough to handle the specialized vocabulary and potentially multilingual nature of space missions.

## Project Overview

For this project, you will develop a program that takes in a string of characters and predicts the next character.
This repo contains a starter submission, including a dummy program that simply generates 3 random guesses.

## Deliverables and Grading

Submit your deliverables on LEARN. Your proposal/reports must have the names `proposal.pdf` and `report.pdf`. Your code must have the name `submit.zip`.

### Initial Proposal (due Feb 2)
**Format**: 1 page PDF.
**Content**:
- What data will you use?
- What model will you consider?
- How will you design your system?

**Grading**:
- 0: Does not answer required questions.
- 1: Answers required questions.

### Midterm Check-in (due Feb 23)
**Deliverable**: Submission of source code and Dockerfile (see "Submitting your project").

**Grading**:
- 0: Program does not execute.
- 1: Program executes.
- 2: Program executes and achieves satisfactory performance compared to held-out naive implementation.
- 3: Program executes and achieves and significantly outperforms naive implementation.
- 4 (Bonus): Program achieves top-3 performance.

**Post-Midterm**: After the midterm, you can make a non-mandatory submission every week. We will automatically evaluate your program on the validation set and make public the evaluation results.

### Final Report (due end of term)
**Format**: 2 pages PDF.
**Content**:
- What is your system design?
- What data did you use? How did you get the data?
- What challenges did you face?
- What experiments did you run? What were the results?
- **Note**: No appendix allowed.

**Grading**:
- **Report**: Out of 10.
- **System Performance**:
    - 0: Program does not execute or does not achieve satisfactory performance compared to held-out naive implementation.
    - 1: Program executes and achieves satisfactory performance compared to held-out naive implementation.
    - 2: Program executes and achieves and significantly outperforms naive implementation.
    - 3 (Bonus): Program achieves top-3 performance.

## Input format

`data/raw/open-dev/input.txt` contains an example of what the input to your program will look like.
Each line in this file corresponds to a string context, for which you must guess what the next character should be.

## Output format

`data/raw/open-dev/pred.txt` contains an example of what the output of your program must look like.
Each line in this file correspond to guesses by the program of what the next character should be.
In other words, line `i` in `data/raw/open-dev/pred.txt` corresponds to what character the program thinks should come after the string in line `i` of `data/raw/open-dev/input.txt`.
In this case, for each string, the program produces 3 guesses.


## Implementing your program

`src/n_gram_lm.py` contains the reference training/testing entrypoint used in this repository.
During training, your may want to perform the following steps with your program:

1. Load training data
2. Train model
3. Save trained model checkpoint

During testing, we will give your program new (e.g. heldout, unreleased) test data, which has the same format as (but different content from) `data/raw/open-dev/input.txt`.
In this case, your program must

1. Load test data
2. Load trained model checkpoint
3. Predict answers for test data
4. Save predictions

`data/raw/open-dev/pred.txt` contains an example predictions file for `data/raw/open-dev/input.txt`.

## Training

Train your model with:

```bash
python3 src/n_gram_lm.py train --work_dir work
```

By default, `src/n_gram_lm.py` reads labeled bootstrap text from `data/raw/open-dev`, additional corpus text from `data/processed/common` and `data/raw/domain`, and weakly labeled examples from `data/raw/train/train.csv`.

### Apollo 11 Dialogue Corpus Note

`data/raw/domain/en/apollo11.txt` is derived from the Apollo 11 transcript hosted by NASA NTRS:
https://ntrs.nasa.gov/citations/20160014392

The original transcript content was parsed and cleaned with ChatGPT to keep dialogue-only text for language-model training.

### Wikipedia Pipeline (Download + Parse + Ingest)

For multilingual augmentation, you can build per-language Wikipedia text shards directly into `data/processed/common`:

```bash
python3 scripts/build_wikipedia_corpus.py \
  --languages en ru zh ja hi ar ko fr de it \
  --page_list data/raw/seed/wikipedia_space_pages.txt \
  --output_root data/processed/common
```

This script:

- Uses a fixed list of seed page titles (default source language: English).
- Resolves language-specific page titles via Wikipedia language links when possible.
- Downloads those exact pages from each target language.
- Normalizes/splits text into training-sized lines.
- Writes `data/processed/common/<lang>/wikipedia.txt`, which is automatically consumed by `src/n_gram_lm.py`.

Useful options:

- `--page_list <path>` points to your own newline-delimited page-title list.

Let's walk through how to use my program. First we will train the model, telling the program to save intermediate results in the directory `work`:

```
python3 src/n_gram_lm.py train --work_dir work
```

After training, a checkpoint is saved to `work/n_gram_lm.checkpoint`.
Next, we will generate predictions for the example data in `data/raw/open-dev/input.txt` and save it in `pred.txt`:

```
python3 src/n_gram_lm.py test --work_dir work --test_data data/raw/open-dev/input.txt --test_output pred.txt
```

## Evaluating your predictions

We will evaluate your predictions by checking it against a gold answer key.
The gold answer key for the example data can be found in `data/raw/open-dev/answer.txt`.
Essentially, your guess is correct if the correct next character is one of your guesses.
For simplicity, we will check caseless matches (e.g. it doesn't matter if you guess uppercase or lowercase).

Let's see what the model gets:

```
python grader/grade.py output/pred.txt data/raw/open-dev/answer.txt --verbose
```

You should see a detailed answer, as well as a success rate.


# Submitting your project
 
 To facilitate reproducibility, we will rely on containerization via Docker.
 Essentially, we will use Docker to guarantee that we can run your program.
 If you do not have Docker installed, please install it by following [these instructions](https://docs.docker.com/get-docker/).
 We will not be doing anything very fancy with Docker.
 In fact, we provide you with a starter `Dockerfile`.
 You should install any dependencies in this Dockerfile you require to perform prediction.
 
 When you submit the project, you must submit a zip file `submit.zip` to LEARN that contains the following:
 
 ```
 src  # source code directory for your program
 work  # checkpoint directory for your program, which contains any intermediate result you require to perform prediction, such as model parameters
 Dockerfile  # your Dockerfile
 team.txt  # your information
 pred.txt  # your predictions for the example data `data/raw/open-dev/input.txt`
 ```
 
 In `team.txt`, please put your name and NetID in the following format:
 
 ```
 Name1,WaterlooID1  # e.g. Victor Zhong,vzhng
 ```
 
 Your `src` directory must contain a `predict.sh` file, which, when executed as `bash src/predict.sh <path_to_test_data> <path_to_predictions>` must write predictions to the file `<path_to_predictions>`.
 For reference, the `predict.sh` file for this example project is in `src/predict.sh`
 Although we encourage you to follow conventions in the example project, it is not necessary to do so.
 *What is necessary is that `predict.sh` generates predictions according to spec*.
 
 On our end, we will extract this zip file and run the following command inside the unzipped directory.
 You should make sure that this works on your end.
 For example, you may consider going to a new directory, unzipping your submission zip file, and running the same command using the example data (e.g. set `<path_to_test_data>` to `$PWD/data/raw/open-dev`).
 
 ```
 mkdir -p output work
 docker build -t cs498-proj/demo -f Dockerfile .
 docker run --rm -v $PWD/src:/job/src -v $PWD/work:/job/work -v <path_to_test_data>:/job/data -v $PWD/output:/job/output cs498-proj/demo bash /job/src/predict.sh /job/data/input.txt /job/output/pred.txt
 ```
 
 If you are curious what these flags in `docker run` mean:
 
 - `--rm` remove container after running
 - `-v a:b` mount `a` in host machine (e.g. path on your machine outside docker) to the container (e.g. path in the docker container).
 
 Running this command will produce `output/pred.txt`, which we take to be your predictions on the heldout test data.
 We will then evaluate your success rate against the heldout answer key using `grader/grade.py`.
 
 Your performance will contain two metrics obtained on the heldout test data, the first being the success rate, the second being the run time.
 Your run time is calculated as the time it takes us to run the `docker run` command.
 
 For reference, the script `submit.sh` will package this example project for submission.
 For convenience, we have put the command we will likely run to grade your assignment in `grader/grade.sh`.
 You should be able to emulate the grading using the example data by running `bash grader/grade.sh example`, which will write your results to the `output` directory.
 
 
 ## Questions
 As more questions come in, we will create a FAQ here.
 In the mean time, please post your questions to Piazza with the tag `Projects`.
 
 Can we work in teams?
 > No.
 
 Will we get help in selecting datasets? Could we also get tips on finding good corpora?
 > Not at this time. Choosing and preprocessing your data is a key component of practising NLP in the real world and we would like you to experience it first hand.
 
 Is there a max processing time limit?
 > Yes, tentatively 30 minutes to run the test data.
 
 Is there a maximum size limit for the program?
 > Yes, tentatively 1 MB for `src` and 1 GB for your checkpoint
 
What does it mean the astronaut "speaks at least one human language"? Will system support mix language .. like mix of English + a different language?
> Your program may receive input that is not English. The input may be a mix of English and other languages.

Can test sentences be virtually anything? e.g. they intentionally test robustness
> Test sentences will not be adversarial. They will be reasonable dialogue utterances.

Do we have unlimited appendix for final report? Such as a 1 page report with a 10 page clarification?
> No. There will be no appendix.
