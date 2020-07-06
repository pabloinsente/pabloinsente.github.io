---
title: Introduction to Jupyter Notebooks - set-up, user-guide, and best practices
published: true
mathjax: true
---

<iframe src="https://github.com/sponsors/pabloinsente/card" title="Sponsor pabloinsente" height="225" width="600" style="border: 0;"></iframe>

**Notes**: 
- This tutorial contains video-lessons at the end of each section
- The Jupyter Notebook version can be found in my GitHub [here](https://github.com/pabloinsente/intro-sc-python/blob/master/notebooks/intro-jupyter-ide.ipynb)

## IDEs: Integrated Development Environments

There are several ways in which we can interact with Python:

1. Via the terminal in Python interactive mode
2. Via the terminal by running Python scripts written in a text editor
3. Via an Integrated Development Environment (IDE)

In this mini-workshop I'll focus on IDEs, as they are more commonly used alternative for data analysis and scientific computing with Python. Pretty much all educational content and examples you will find on-line about the subject are created in and for IDEs. 

Popular IDEs for Python are Spyder, PyCharm, VSCode, and Jupyter Notebooks. This are all valid options with advantages and disadvantages. I'll focus in Jupyter Notebooks for a couple of reasons: 

1. They can be run in cloud-based computing environments easily 
2. They allow to create interactive documents, combining text, code, and graphics, which is great for educational content
3. Most on-line examples and educational materials in scientific computing and data analysis are created in Jupyter Notebooks
4. It is beginner friendly and intuitive to use

Jupyter Notebooks do have several drawbacks that can become a problem for more advance users, which may prefer to work in text-based IDEs like PyCharm or VSCode. I will briefly mention such drawbacks later, so you are aware of them. It is also worth notice that VSCode and PyCharm allow for Jupyter-like interfaces, which you can check it our [here](https://code.visualstudio.com/docs/python/jupyter-support) and [here](https://www.jetbrains.com/help/pycharm/jupyter-notebook-support.html).

## Jupyter Notebooks

Jupyter is an [open-source software](https://github.com/jupyterlab/jupyterlab) for interactive computing for a variety of programming languages like Python, Julia, R, Ruby, Haskell, Javascript, Scala, and [many others](https://github.com/jupyter/jupyter/wiki/Jupyter-kernels). The document you are reading right now is an example of a Jupyter Notebook. 

Jupyter Notebooks have become popular among researchers and data scientists because of several convenient characteristics:

1. Allows for the combination of a narrative, code, and the results of computations in one place
2. Easy to use intuitive interface
3. Flexibility on supporting multiple programming languages
4. Integration with cloud computing environments

Such characteristics permit to effortlessly reproduce the workflow of most researchers and data scientists: the description of a research problem and methods in prose, plus the code to run the analysis and models to analyze datasets, and the presentation of results in tables and charts. You can even export your computational Notebook into PDF, Markdown, HTML, and others easy to share formats.

Jupyter Notebooks evolved from a project called [IPython](https://ipython.org/), which was created by the Colombian Berkeley Professor [Fernando Pérez](https://bids.berkeley.edu/people/fernando-p%C3%A9rez), who at the time was a graduate student in Physics at CU Boulder. [Here](https://www.youtube.com/watch?v=xuNj5paMuow) you can watch a Fernando's presentation about his journey creating Jupyter Notebooks.

Although nowadays Jupyter is the most widely used IDE for scientific computing and data science worldwide, the idea of computational notebooks predates Project Jupyter. The first computational notebooks, today named [Wolfram Notebooks](https://www.wolfram.com/notebooks/), were introduced by [Stephen Wolfram](https://www.stephenwolfram.com/) for the [Wolfram Mathematica](https://www.wolfram.com/mathematica/) programming language. The issue was that Wolfram Mathematica and Wolfram Notebooks are a closed-sources proprietary software, i.e., you have to pay for it. 

Fortunately, today we have access to Jupyter Notebooks for free, which is developed and maintained primarily by a large community of users from all over the world.

## JupyterLab 

[JupyterLab](https://github.com/jupyterlab/jupyterlab) is the next-generation interface for Jupyter Notebook. Essentially, they are an extension build on top the classic Jupyter Notebook, but with improved capabilities and features. I will use the JupyterLab interface for this mini-workshop, as it is the most up to date version of Project Jupyter, and is expected to fully replace the classic Jupyter Notebook in the short term. Project Jupyter developers advise the use of JupyterLab as they are investing they efforts on maintaining and developing this platform.

Note that there some minor but important differences between the interface Jupyter Notebooks and JupyterLab, so you are advised to search for JupyterLab specific extensions and tutorials, since functionality may differ. 

## JupyterLab basics

### Installing JupyterLab

Before installing JupyterLab, I am assuming you have a recent version of Python 3 installed. Any Python version greater than 3.6 should work. Utilizing a virtual environment it is also recommended to isolate your JupyterLab installation.

 It is also good idea to update `pip` before installing jupyterLab by running:

```bash
python -m pip install --upgrade pip
```

JupyterLab can be installed with `pip` in the terminal as:

```bash
# is recommended to run this in a virtual environment
pip3 install jupyterlab
```

or as:

```bash
python3 -m pip install jupyterlab
```

A second option is with `conda` 

```bash
# is recommended to run this in a conda virtual environment
conda install -c conda-forge jupyterlab
```

I personally prefer to use `pip` as I find it's simpler, cleaner, and works out-of-the-box with your Python installation.

Also take into account that the above instructions will install the latest stable release of JupyterLab. 

To check you installation was succesful, run: 

```bash
jupyterlab --version
```

### Launching JupyterLab

To launch JupyterLab open the terminal, navigate to your working directory, and run:

```bash
jupyter lab
```

JupyterLab will launch a session in your default browser. If you want to launch JupyterLab in a different browser, you can either change your default browser or to copy-past the Notebook address in your desired browser. 

### JupyterLab interface

The JupyterLab interface consists of: 

1. a **main work area** containing tabs for the notebooks, terminals, and text files
2. a **collapsible left sidebar** containing a file browser, the running kernels and terminals, the command palette, the cell inspector, the list of open tabs, and any other extension you may have activated.
3. a top **menu bar**. 

### Creating and renaming a Notebook

JupyterLab will open the Launcher tab as default, where you can select what kind of instance you want to run. By default, JupyterLab will allow for a Notebook with Python 3 kernel, a IPython console, a bash terminal, a Markdown file, and a text editor. 

To create a Notebook click under "Notebook" in the Launcher, and select a Python 3 kernel when prompted. Alternatively, you can create new Notebooks by clicking "File" in the top menu bar, then "New", and then "Notebook. 

It is important to rename the Notebook as JupyterLab will give an "Untitled.ipynb" name as default to all new Notebooks. 

## Video I - Introduction and set-up

<div class="embed-container">
<iframe width="560" height="315" src="https://www.youtube.com/embed/U5Hg1Anxy7g" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

### Interacting with files

JupyterLab will load up directory and subdirectories from which you open it up in the terminal. As JupyterLab is very flexible, you will be able to open pretty much any file by double clicking on it: notebooks, images, text files, json files, csv files, and much more. 

One of the improved capabilities of JupyterLab is its ability to open large csv files in a excel-like nice looking interface. 

Another great JupyterLab feature is the possibility of rearranging your workspace with multiple windows at once, which is great for when you want to work with multiple files side by side. 

Additionally, files that can be open in more than one format, can be put side by side, and the changes you make in one view/format of the file will reflect in the other view/format. For instance, Markdown files can be open as rendered versions or as raw text. Notice that you have to save the changes before they are reflected in the other view/format. 

A common task in data manipulation is to open files like `csv` datasets and `png` images. To import such files into your Notebook, you will need to know the "path" to the file. File paths can be easily obtained by right-clicking in the desired file and selecting "Copy path". 

Uploading files can be done by clicking the "Upload Files" arrow-icon in the left sidebar or by dragging and dropping the files onto the file navigator in the browser. 

Downloading files can be done by right-clicking in the desired file, and selecting the "Download" option. 

## Interacting with Notebooks

As I mentioned earlier, Notebooks main advantage is its capacity to combine a narrative (text), with code to run analysis and models, and the analysis outcomes as tables and plots, all in one place. As a bonus, Notebooks allow to render LaTeX output, which is fabulous if you need to introduce equations.

Notebooks are made out of collections of **cells**. A cell is simply a rectangular box in which you can type and visualize stuff. There are three types of cells: 

- **Markdown cells**: this are used to write your document, and to organize the content in titles, sub-titles, and so on
- **Code cells**: this are used to run code
- **Raw cells**: this are essentially plain text

The easiest way to change between cell types, is by clicking on the drop-down tab at the top of the Notebook and selecting the one you want. You can also use the keyboard shortcuts which are `Esc` + `M` for Markdown, `Esc` + `Y` for Code, and `Esc` + `R` for Raw. 

To run a cell, you can either go to the top menu bar, select "Run", and then click in "Run Selected Cell". In practice, most people uses the `Ctrl` + `Enter` keyboard shortcut. You can also use `Shift` + `Enter` or `Alt` + `Enter` to run the current cell and insert a cell below. 

You can create new cells by clicking in the `+` symbol in the top bar of the Notebook, or with the `Esc` + `B` keyboard shortcut. 

To delete cells, you can right-click on the cell and select "Delete Cell", or you can use `Esc` + `DD` shortcut. 

To copy cells, you can right-click on the cell and select "Copy Cell", or to use `Esc` + `C` keyboard shortcut. To past a copied cell, you can right-click on the cell and select "Paste Cell Below" to insert the cell below th active cell, or to use `Esc` + `V` keyboard shortcut. 

To cut cells, you can ither use the "scissors" icon at the top of the workspace bar, or to use the `Esc` + `X` keyboard shortcut. 

When you insert new cells, you won't be able to type content into the cell immediately. You need to enter in "command" or "enter mode". To do this you can double-click on the cell, or to use the `Enter` keyboard shortcut. 

Accidentally deleting cells is something may happen often in Notebooks. To recover a deleted cell, you can right-click on the cell that became active after deletion, and select "Redo Cell Operation", or to use the `Shift` + `Z` keyboard shortcut. 

Remembering Python syntax, commands, and library methods, can be thought at the beginning. Notebooks offer a few useful tools to help with this. When typing  a Python command, you can click `Tab` and JupyterLab will generate a drop-down menu where you can search for commands and methods to complete your code. This is the so-called *Tab completion*. Tab completion is usually very powerful in IDEs like VSCode and Pycharm, but a bit slow and incomplete for JupyterLab.

A second Notebook tool to remember commands and methods is the "Contextual Help". The contextual help is activated by clicking on "Help" in the top bar menu, and selecting "Show Contextual Help", or wit the `Ctrl` + `I` shortcut. The Contextual Help will open as a new tab, so you can rearrange your workspace to have the Contextual Help tab side by side your Notebook. Contextual Help works by searching the documentation for a function when you select it. Keep in mind it won't work for all functions. 

To move cells around you can hover over the left-side of the cell, and click to drag and drop the cell in a new location. Although this is a nice feature, I strongly advise against it for reasons I will review later.

To save changes made to your Notebook, you can click on "File" at the top bar menu and select "Save Notebook" or "Save All", or to simply use the `Ctrl` + `S` keyboard shortcut. Fortunately, JupyterLab routinely save your progress in the background so it can be recovered if your browser closes for any reason. 

## Video II - Intercting with files and Notebooks

<div class="embed-container">
<iframe width="560" height="315" src="https://www.youtube.com/embed/pQsDd0N2kNQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## Interacting with the Kernel

The first time I heard about a "Kernel" I was deeply confused (I'm still a bit confused to be honest). Put simply, a Kernel is an "instance" of the Python interpreter that runs in the background and process the code instructions you type in Notebook cells. Same for any other dynamic programming language like R, Scala, or Julia. This is why "Kernels" of multiple programming languagues can be added and run in parallel in JupyterLab. 

You can see what Kernels are being run in your session by clicking on the "Terminals and Kernels" icon in the left sidebar. Such menu also allows to shut down Kernels. Notice thay closing a Notebook or Console will not shut down the running Kernel. Shutting down a Kernel must be explicitly done in the Kernel session menu or the terminal where you open up JupyterLab.

The "Kernel" tab in the top bar menu also allow to do Actions on Kernels, like shutting down the current Kernel or all running Kernels.

Common Kernel operations are Interrupting and Restarting. Interrupting is sometimes necessary to stop computations have been running for too long and may be corrupted. Restarting is sometimes necessary to generate a clean environment to work with the same Notebook. Interrupting can be done in the "Kernel" menu or with the `Esc` + `II` keyboard shortcut. Restarting can also be done in the "Kernel" menu or with he `Esc` + `OO` menu.

Another important operation is to "Restart the Kernel and Run All Cells". This is often done because you want to make sure the code in the Notebook run without problems from top-to-bottom, which can be a problem given how Notebooks work. This can only be done in the "Kernel" menu. 

## JupyterLab Extensions

JupyterLab has the flexibility of incorporating extensions, this is, additional functionality which is not available "out-of-the-box". Such extensions are primarily created by the community of users and developers in the Jupyter community, and are free to use. 

Under the hood, JupyterLab is essentially a bunch of JavaScipt code, which is the dominant programming languague in the web development sphere. This means that JupyterLab extensions are developed in JavaScript, which implies to will need Node.js in your machine to install JupyterLab extensions.

There are two ways to install extensions: with the Extension Manager and in the terminal. The Extension Manager it is also an extension (very meta), which provides a graphic user interface within JupyterLab to install extensions. You won't find it at first because it is disable by default. To enable it, go to the search bar in the command pallet, and search for "Extension Manager". Once enabled, you will see a puzzle-shape icon in the left sidebar. 

You can search and install extensions by searching by name in the search bar in the Extension Manager. Now, before you jumping to install extensions, beware that extensions allow for the execution of unchecked code in your Kernel and browser instance. JupyterLab partially address this issue by adding a JupyterLab Icon on the extensions "verified" by Project Jupyter. I have to acknowledge that I have never paying attention to who created the extension or if its verified or not before installing it. You may be wiser than me and check that out before installing extensions.

Installing extension in the Extension Manager is as simple as to click on the "Install" icon. Once the installation is done, JupyterLab will prompt you to reload your workspace to make the extension available for use. Disabling extensions can be done in the Extension Manager as well, by simply searching the extension in the search bar and clicking on "Disable". 

You can also install extension in the command line. To do this, you have to run :

```bash
jupyter labextension install my-extension
```

Where "my-extension" stands for the extension name in the [npm package repository](https://www.npmjs.com/). For instance, to install the "@jupyterlab/toc" extension:

```bash
jupyter labextension install @jupyterlab/toc
```

You can search for extensions [here](https://www.npmjs.com/) by searching for "jupyterlab" in the search bar. 

Note that if you are using a virtual environment to run JupyterLab, you must open the terminal, activate the virtual environment, and then install the extension. Once the installation process is done, you will have to reload JupyterLab by clicking in the reload icon in your browser.

## Cool and useful JupyterLab extensions

JupyterLab benefits from a constantly growing library of extension to enhance functionality and user experience. Here I'll just mention five of my favorites that I found significantly improve my productivity and enjoyment.

### Table of contents extension (TOC)


The TOC extension allows to organize sections in a Table of Contents, which is auto-generated following markdown heading conventions. You can click and navigate the document with the TOC in the left sidebar.

To install `toc` extension the terminal run:

```bash
jupyter labextension install @jupyterlab/toc
```

Or search for it in the Extension Manager

### Spell checker extension

I happen to write a lot in JupyterLab, and one of my major complaints is the lack of an spell checker as traditional text editors. The jupyterlab_spellchecker extension partially address this by highlighting misspelled words. It won't suggest the correct spelling, but at least it will let you know you misspelled something.

Note that this extension will also highlight any unrecognized word according to the [Typo.js](https://github.com/cfinke/Typo.js) library, meaning that words like "JupyterLab" will be highlighted. It also can be a bit annoying at first to have misspelled words highlighted as it does it while you write, instead of after. Yet, as non-native speaker, I prefer to tolerate such annoyance to produce better writing.

To install in the terminal `jupyterlab_spellchecker` run:

```bash
jupyter labextension install @ijmbarr/jupyterlab_spellchecker
```

Or search for it in the Extension Manager.

### Collapsible headings

Sometimes Notebooks may get very long, which makes navigation annoying or slow. There are sections you may not even need to work anymore and that would be better to have out of your sight. The `collapsible_headings` extension allows to collapse / uncollapse sections by clicking on the caret icon on the top left corner of the cell defining a section.

To install the `collapsible_headings` with the terminal run:

```bash
jupyter labextension install @aquirdturtle/collapsible_headings
```

Or search for it in the Extension Manager.

### Shortcut manager

As you may have notice by now, JupyterLab has a wide variety of keyword shortcuts to improve productivity. The `shortcutui` allows to both explore the current available shortcuts in your system, and to define new ones, in a simple graphic interface. 

To access the keyboard shortcuts editor click on the "Settings" or "Help" tabs in the top menu bar, and select "Keyboard Shortcut Editor".

To install `shortcutui` with the terminal run:

```bash
jupyter labextension install @jupyterlab/shortcutui
```

Or search for it in the Extension Manager.

### Jupyterlab Neon Theme

Some people like light-themes (white background) editors, some like dark-theme editors, and some, like me, like shinny neon purple editors. It just easier on my eyes and pleasant to look at. The Jupyter Neon Theme extension provides exactly that. 

To install the `jupyterlab_neon_theme` extension in the terminal run:

```bash
jupyter labextension install @yeebc/jupyterlab_neon_theme
```

To actually use it, click on "Settings" in the top bar menu, then in "JupyterLab Theme", and then in "JupyterLab Neon Theme" (or any theme you have available and want to try).

Or search for `jupyterlab_neon_theme` in the Extension Manager.

## Video III - Kernel and Extensions

<div class="embed-container">
<iframe width="560" height="315" src="https://www.youtube.com/embed/lYFusU11RbY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## Exporting Notebooks

Notebooks have the `.ipynb` extension, which stands for interactive python notebook. However, Notebooks can be exported in a wide variety of formats:

- Asciidoc .asciidoc
- HTML .html
- Latex .tex
- Markdown .md
- PDF .pdf
- ReStructured Text .rst
- Executable Script .py
- Reveal.js Slides .html

To access to the export options click on "File" -> "Export Notebook As" and select the desired format. 

### Exporting to PDF and LaTeX

To export Notebooks to PDF and LaTeX formats you will a need couple of packages first. In particular, `pandoc`, `nbconvert` and `TeX`. The installation process for those packages will differ depending on your Operating System, i.e., whether you are running Linux, macOS, or Windows.

Instructions to install `pandoc` can be found [here](https://pandoc.org/installing.html)

Instructions to install `nbconvert` can be found [here](https://nbconvert.readthedocs.io/en/latest/install.html)

Instructions to install `TeX` can be found [here](https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex)

### Recommended formart to export and share with others

When I started using Jupyter and wanted to share a non-interactive version of my Notebook, my first instinct was to generate a PDF. Turns our, that since Notebooks are all JavaScript under the hood, it is pretty simple to generate a nice and consistent looking HTML file. Any person you send a HTML file will be able to open the file by just double clicking on it as long they have a browser, which every computer has. 

I highly recommend to export Notebooks to share in HTML format as they may save you a lot of pain trying to export Notebooks in PDF properly formatted. 

## Running Notebooks in VSCode

VSCode is a free and open-source general purpose multi-platform (i.e., Linux, Windows, and macOS compatible) text IDE developed by Microsoft. Nowadays is one of the most popular IDEs and text-editors among programmers and data scientist. 

VSCode has recently incorporated the capacity to open `.ipybn` extension files, i.e., Notebooks, providing a Notebook-like interface. This is certainly a good alternative to try out if you are not fully comfortable with the web browser interface. I personally do not use it as I found out that rendering text, LaTeX formulas, and pictures is cleaner and easier in the browser interface.

To install VSCode go to the [official website](https://code.visualstudio.com/) and follow the installation instructions for your system.

[Here](https://code.visualstudio.com/docs/python/jupyter-support) is the official VSCode guide to run Jupyter Notebooks within VSCode.


## Video IV - Exporting and VSCode

<div class="embed-container">
<iframe width="560" height="315" src="https://www.youtube.com/embed/-jzEB34a0RE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## Notebooks weaknesses

Until this point, we have highlighted the many awesome capabilities of JupyterLab and Notebooks. However, Notebooks have a series of weaknesses to be mindful about and to consider, such that you can do two things: 

1. Decide when is a good idea to use a Notebook
2. If you decide to use, to take the proper precautions to prevent problems

Here is a brief list of issues that myself and others [on the Internet](https://www.youtube.com/watch?v=7jiPeIFXb6U) have mentioned about Notebooks:

- Out of order execution
- Source control is hard
- Dependency management is hard
- Modularization is hard
- Testing is hard
- Code reviews are hard
- Code is hard to extend
- Refactoring code is hard
- Maintaining code is hard
- Code collaboration is hard
- Encourage poor programming practices

I will not delve into all of this topics in consideration of time and space, and because many of this issues are actually problematic for large scientific projects rather than for small data analysis or modeling procedures. I will focus in just three issues: **out of order execution**, **modularization**, and **refactoring** and **debugging**. 

### Out of order execution

When you write code for an analysis, the code must run in the exact same sequence to yield the same results or to even run at all. Traditional code written in plain text-editors and then run in the terminal, will always do this by definition. Notebooks, on the other hand, make very easy to run cells out of order. It has happened to me dozens of times during the ~3 years I have been using Notebooks, and it is a common complaint on the Internet among data scientists.

Most of the time, out of order execution will not matter that much. You will realize early and fix it. But once in a while, you may introduce a "bug" or mistake in your code than can cost you hours of detective work. That is the best case scenario. Worst case scenario, you never realize your mistake and your results will be wrong without you knowing it.

For instance, imagine you have to run these three steps:

1. Get rid of NA in your data
2. Computer the weighted average of two variables
3. Create a new variable based on those two variables

Several things may go wrong here. First, you may forgot to run the cell cleaning the NA values. Second, you may compute the average first, and run the cell to eliminate the NA values second. 

Although making such mistakes may sound unlikely, at least in mat experience, is not. Here is a typical scenario: you write the code for cleaning the NA and computing the average on different cells, so everything is nicely organized. Then you go for a coffee and get distracted. Then you come back to your computer, your cursor is on the cell to compute the average, so run that, and keep working. Afterwards, you may spend a good couple of hours writing and running more code, and writing analysis from your results, just to realize several hours later that you forgot to run the cell to clean the NA values. Too bad.

Here is another one: you suddenly realize you made a mistake in the cell to clean the NA values. You go back, change the cell, and rerun the cell. But ops! you forgot to rerun the previous cell to reload the data into your Notebook, so now your cleaning process is mess up. You go back and rerun that cell. Then you get distracted with your dog barking. Then you go an run the average cell. You get distracted again with an email. Then you go back and remember to run the cell to clean the NA values. There you go: you ran the cells out of order. Again, this simply can't happen with a script, since the script will rerun everything in order from top to bottom each time. 

Even though you may think that you focus is too good to make such mistakes, it is often the case that people get chronically distracted with working on computers with Internet access. **So, be careful**: if you really need to run long sequences of code, you may consider to switch to a text-file instead of a Notebook.

### Modularization is hard

When your code start to grow, a common recommended best practice is to divide your code into **modules**. This is simply having separate text-files with code to do **one task**. For instance, a large project may have scripts to: (1) load the data, (2) pre-process the data, (3) run descriptive statistics and plots, (4) run statistical models and generate the output. 

Doing all of this with Notebooks is not impossible, but hard. It is just way too easy to get trapped in the flow of writing everything in a single Notebook. 

### Refactoring and debugging is hard

Refactoring refers to the practice to rewrite and reorganize your code for readability and performance. Debugging refers to the exercise of searching for mistakes in your code. 

**When code modularization is hard, refactoring and debugging are hard**. When code is modularized into single-purpose isolated scripts, it is relatively easy to focus in ways to rewrite such code to be clearer and faster to run. This is hard with Notebooks. Debugging code is also relatively easy when code is modularized, and refactored. If everything is dumped into a Notebook with 100 cells, finding where you made an error in your code can be a nightmare. 

## Video V - Notebooks weaknesses

<div class="embed-container">
<iframe width="560" height="315" src="https://www.youtube.com/embed/lMlN-2W1rxE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## When to use Notebooks

The purpose of make Notebooks weaknesses visible, is to encourage you to think carefully about when to use Notebooks, and to take precautions. 

In my view, Notebooks are a good option, sometimes the best option, in the following scenarios: 

- Relatively short data analysis or modeling task
- Task that rely more in the narrative than in the code itself, which is short
- Task that are primarily about data visualization, as Notebooks can render plots and images easily
- Tutorials and educational content
- Analysis that entail lots of equations 
- Task that require relatively "small" or "medium" size datasets
- Task that are heavily dependent on constant interaction with the results of running your code, i.e., highly interactive development
- Prototyping code that will be eventually transfered to a script

People differ in their appreciation of Notebooks. Only you and your experience using them can inform what is best for you.

## Basic good practices to work with Notebooks

Now that we covered Notebook weaknesses, it is a good idea to review a few "good" practices when working with Notebooks such that you can prevent as many mistakes as possible.

My recommendations are mix of my own experience, things I have learned from other Notebook users, and [this article](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007007#sec004) about 10 rules to work with computational Notebooks.

Here is the list:

1. Cells should do one task at the time
2. Document your coding process
3. Restart the Kernel and rerun all the cells often
4. Save and use version control often
5. Modularize your code as much as possible
6. Document your dependencies
7. Switch to a script if your Notebook grows too long

### Cells should do one task at the time

It is tempting to write code cells that run multiple lines of code and tasks all at once. This will probably make hard to read, document, and debug code cells if something goes wrong. Ideally, you want cells to do one task at the time. One task may involve multiple lines of code, but it is still one task. 

For instance, computing the mean between two list of integers is a perfectly fine task to have in one cell as:


```python
from statistics import mean 
```


```python
list_a = [1, 2, 3]
list_b = [4, 5, 6]
list_c = list_a + list_b
mean = mean(list_c)
print(f'mean of list-a + list-b = {mean}')
```

    mean of list-a + list-b = 3.5


Now, it is not always clear what "a task" entails. What if you define "the task" as computing descriptive statistics as the mean, median, and mode? Should all three go into a single cell or three separate cells? Since computing those is so simple, I would put everything in once cell, and move the print statements onto a separate cell as:


```python
from statistics import mean, median
```


```python
list_a = [1, 2, 3]
list_b = [4, 5, 6]
list_c = list_a + list_b
mean = mean(list_c)
median = median(list_c)
max_v = max(list_c)
```


```python
print(f'mean of list-a + list-b = {mean}')
print(f'median of list-a + list-b = {median}')
print(f'max value of list-a + list-b = {max_v}')
```

    mean of list-a + list-b = 3.5
    median of list-a + list-b = 3.5
    max value of list-a + list-b = 6


If instead of simple calculations, the task were to run several statistical models like Principal Component analysis and Logistic Regression, I would definitely separate everything.

### Document your coding process

Documenting code is a whole art in itself. Yet, there a few things you can implement right away to make code more reliable. Take the code from out previous example. There are two things we can do to improve upon it: (1) to use markdown cells to describe what are doing, (2) to add comments in along with the code.

Now, for such a short and simple task is often recommended to not add in-line commentaries as the code is self-descriptive. However, I'll add comments just for the sake of example.

Something like: 

#### Descriptive statistics

Here we compute the mean, median, and max value for our lists.

```Python
list_a, list_b = [1, 2, 3],[4, 5, 6] # assign list to variables for further computation
list_c = list_a + list_b # join list to compute descriptive stats 
mean = mean(list_c) 
median = median(list_c)
max_v = max(list_c)
```

In general, you want to comment on the "why" instead of only the "what" of a line of code.

### Restart the Kernel and rerun all the cells often

Just restarting your Kernel and rerun cells often it is a tremendously effective strategy to avoid the out-of-order execution problem. Since i started to do this the mistakes in my code where reduced dramatically. Quick and simple.

### Save and use version control often

Although JupyterLab will constantly save your progress, clicking `Ctrl` + `S` often and saving your progress in a remote version control system as Git/GitHub is a life saver for when things go wrong with Notebooks. Version control system is a topic in itself, but in short, it is a way to save copies of your work at different points in time in the cloud. Such "images" or "clones" of your work can be accessed and recovered easily in case you made some serious mistake in your code.

### Modularize your code as much as possible

True, I previously argued that modularization was hard in Notebooks. But hard does not mean equal impossible. As you become a more proficient coder, you will fine ways to refactor and modularize your code in manner that will make work with Notebooks more reliable and cleaner. 

### Document your dependencies

Dependencies are all the software and libraries you used in your project. For instance, your software stack for a project may look as:

- python==3.6.8
- jupyterlab==2.0.1
- numpy==1.18.2
- pandas==1.0.3
- pip==20.0.2

Documenting that at the top of your Notebook is a good way to make clear to your future self and others potential users of your code, what are the requirements to reproduce your analysis. 

People have created utilities for that like `whatermark`


```python
%load_ext watermark
%watermark
```

    2020-06-11T15:58:30-05:00
    
    CPython 3.6.8
    IPython 7.15.0
    
    compiler   : GCC 7.5.0
    system     : Linux
    release    : 4.19.104-microsoft-standard
    machine    : x86_64
    processor  : x86_64
    CPU cores  : 16
    interpreter: 64bit


### Switch to a script if your Notebook grows too long

No matter how many precautions you take, you will reach a point where using Notebooks it is not the most functional alternative. You can insist on using a Notebook for large projects with many moving pieces, but you will probably start to experience the problems mentioned above. In such cases, a IDE like VSCode, Pycharm or Atom are probable your best option.

## Video VI - When to use Notebooks and Best Practices

<div class="embed-container">
<iframe width="560" height="315" src="https://www.youtube.com/embed/_M2rbm_zh50" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## Final JupyterLab Tips

Here a few additional final tips to use Notebooks effectively:  

- **Command Palette**: if you do `Ctrl` + `Shift` + `C` you will have access to the Command Palette, which is a centralized command system to search and use all JupyterLab utilities. 
- **Create console for Notebooks**: if you right-click onto a cell a select "New Console for Editor", JupyterLab will open a Python interactive terminal at the bottom of the screen. Such instance shares the variables and information you have run already in your Notebook. It is a great way to try out code without having to change the Notebook, and to get richer output when running code. 
- **Use the Help menu**: the "Help" tab at the top menu bar has the official documentation for JupyterLab which comes in handy when you are trying to learn something new or fix something does not work. Googling of course will always help, but sometimes answers to your questions will be outdated and plainly wrong. 

## Resources to learn more

Here you can find a list of resources to learn more about JupyterLab and become and effective user:

- [Ten simple rules for writing and sharing computational analyses in Jupyter Notebooks](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007007#sec004)
- [Jupyterab official documentation](https://jupyterlab.readthedocs.io/en/stable/index.html)
- [JupyterLab: The Next Generation Jupyter Web Interface](https://www.youtube.com/watch?v=ctOM-Gza04Y)
- [JupyterLab: The Evolution of the Jupyter Notebook](https://www.youtube.com/watch?v=NSiPeoDpwuI)
