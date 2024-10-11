---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Eliza


This is a very simple implementation of the Eliza chatbot. It uses a set of rules to generate responses based on what you type. It's not a very sophisticated chatbot, but it's illuminating to look at how it works.

* Weizenbaum, J. (1966) ‘ELIZA—a computer program for the study of natural language communication between man and machine’, *Communications of the ACM*, 9(1), pp.&nbsp;36–45. doi:10.1145/365153.365168.

```python
from eliza import *
```

The rules are in the `rules.yaml` file, and the `read_rules` function will load them from the file.

```python
all_rules = read_rules('rules.yaml')
```

Now we have the rules, you can talk to Eliza. Try talking about your mother or father, or about your dreams.

When you've finished, type `quit` to stop Eliza.

```python tags=["style-solution"]
eliza_loop(all_rules)
```

```python
all_rules
```

```python
match('this is some text'.split(), '?X ?Y'.split())
```

```python
splits('this is some text'.split())
```

It's now time for you to write some of your own rules. Back in the main Jupyter interface, find the `rules.yaml` file in the list of notebooks and other files. Select the checkbox next to the `rules.yaml` file then press the `Duplicate` button at the top of the window. Then click on the `rules-Copy1.yaml` file you just created to open it for editing.


The rules look like this:
```
- pattern: ?X i am glad ?Y
  responses:
  - how have i helped you to be ?Y
  - what makes you happy just now
  - can you explain why you are suddenly ?Y
```
The `pattern` is what matches against what you type in. The parts `?X` and `?Y` are Eliza's variables, where Eliza captures the interesting parts of what you say.

For instance, the phrase "hi i am glad to meet you" will match the pattern "?X i am glad ?Y" with the bindings of `?X` matching "hi" and `?Y` matching "to meet you", like this:

```python
all_bindings = match("hi i am glad to meet you".split(), "?X i am glad ?Y".split())
bindings = all_bindings[0]
bindings
```

(Note that Eliza splits text into lists of words, rather than keeping it as a string, and that `match` returns a list of possible matches.)

Eliza uses those matches to fill out the response:

```python
fill("how have i helped you to be ?Y".split(), bindings)
```

As you can see, Eliza has used what was captured in the `?Y` variable to fill out the response.

Where a rule has several responses, Eliza will pick one of them.

It could be that several rules match what you type. In that case, Eliza will always use the first rule that matches.

<!-- #region tags=["style-activity"] -->
## Exercise
<!-- #endregion -->

<!-- #region tags=["style-activity"] -->
Now you've seen how Eliza works, it's time to write your own rules. Create a new rule in the file `rules-Copy.yaml` that has the pattern "?X bananas ?Y" and a single response "i think bananas ?Y". Make sure the new rules is the first or second one in the file. Save the file when you're done.

(Note that the rules, formatted as YAML statements, are as sensitive to whitespace and layout as Python text. Be sure to use exactly the same formatting as the existing rules.)
<!-- #endregion -->

<!-- #region tags=["style-student"] -->
### Solution
<!-- #endregion -->

<!-- #region tags=["style-student"] -->
The rule should look like this:
```
- pattern: ?X bananas ?Y
  responses:
  - i think bananas ?Y
```
<!-- #endregion -->

<!-- #region tags=["style-student"] -->
## End of solution
<!-- #endregion -->

Now see if your new rule works. Tell Eliza that "bananas are evil" and see what the response is. (Remember to type "quit" to finish.)

```python tags=["style-solution"]
eliza_loop(all_rules)
```

Now load your new set of rules into Eliza and again tell Eliza that "bananas are evil". You should see a different response, based on your new rules.

```python
all_rules = read_rules('rules-Copy1.yaml')
```

```python tags=["style-solution"]
eliza_loop(all_rules)
```

<!-- #region tags=["style-activity"] -->
## Exercise
<!-- #endregion -->

<!-- #region tags=["style-activity"] -->
Now you've seen how the rules are used by Eliza, try writing a few of your own. The rules work best if you focus them on words or phrases like "dream" or "remember". Try writing a couple of rules about what a person may imagine. Put them in the middle of the file, near the existing rules for "dreamed".

Spend no more than **20 minutes** on this.

Write the new rules, save the file, and reload it in this noteobok. Check to see that your new rules are used.

Post your new rules in your tutor group and try a few of the rules suggested by other students. 

Hopefully, what you learn from this process is that writing rules can be a laborious and unforgiving process, and one that's difficult to get right.
<!-- #endregion -->

```python

```
