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

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Language generation and language models
<!-- #endregion -->

Can you give the next word in the following phrases?

> Never gonna give you _
> 
> That's one small step for a man, one _
> 
> A bird in the hand is worth _
> 
> London bridge is falling _
> 
> To be or not to be, that is _
> 
> It was the best of times, it was _
> 
> The quick brown fox jumped _

The chances are, you were able to give the next word, if not complete the whole phrase. This is because, in a lot of cases, language use is stereotyped. Certain words follow from certain phrases, and we can use that to help a machine work fluently with language.

We can use this facility in a few ways. One is to help us understand human speech input. Often, the sounds we make while speaking can have more that one interpretation (try saying quickly "recognise speech" and "wreck a nice beach" to someone, and ask them which is which). In these cases, having some idea of the likely words can help us disambiguate the sounds the machine hears. We can also use the predictability of language to detect spelling and grammar mistakes; a grammar checker can detect incongurous words and suggest them for revision.

Another application, one we'll be looking at here, is about _generating_ text that reads like a plausible new example of some source. If we build a language model using only text from one source (or a limited range of sources), that model will reflect that corpus of text. If we generate text with that model, it should have a similar style to the source.

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## A language model
<!-- #endregion -->

<!-- #region -->
These applications rely on having a _language model_, a description of what the language should look like. There are many types of language model. You've probably heard of the "large language models" used by tools such as ChatGPT. For this example, we'll use a much smaller model, but it's surprising how well even this small, simple model can work.

Our language model is inspired by the quiz above: if we know the last few words that have been used, we can make a prediction about what comes next. This is called an **_n_-gram** model in the literature, where _n_ is now many words of context we're using.

For instance, let's we take the first line of _A Tale of Two Cities_

> It was the best of times, it was the worst of times …

We can build a 2-gram (bigram) model of this text. We slide a two-word-long window along the text and record, for each bigram, the word that comes next. Sliding the window looks a bit like this:

| it | was | the | best | of | times | it | was | the | worst | of | times |
|----|-----|-----|------|----|-------|----|-----|-----|-------|----|-------|
| it | was | the |  |  |  |  |  |  |  |  |  |
|  | was | the | best |  |  |  |  |  |  |  |  |
|  |  | the | best | of |  |  |  |  |  |  |  |
|  |  |  | best | of | times |  |  |  |  |  |  |
|  |  |  |  | of | times | it |  |  |  |  |  |
|  |  |  |  |  | times | it | was |  |  |  |  |
|  |  |  |  |  |  | it | was | the |  |  |  |
|  |  |  |  |  |  |  | was | the | worst |  |  |
|  |  |  |  |  |  |  |  | the | worst | of |  |
|  |  |  |  |  |  |  |  |  | worst | of | times |

We can see that the bigram "it was" occurs twice in that sentence, and both times it is followed by the word "the". We can also see that the bigram "was the" occurs twice, but it is followed by different words each time: once by "best", once by "worst".

The full bigram model from this sentence looks like this:

* it was → the: 2
* was the → best: 1, worst: 1
* the best → of: 1
* best of → times: 1
* of times → it: 1, None: 1
* times it → was: 1
* the worst → of: 1
* worst of → times: 1

With only this short amount of text, the language model doesn't really tell us much interesting. We need more text. If we take the entire first chapter of the book, we find 878 bigrams, most of which occur only a couple of times. The most frequent ones are:

* and a → knife: 1, queen: 2, thousand: 1
* and seventy → five: 3
* and the → fair: 1, farmer: 1, guard: 1, majesty: 1, mob: 1, musketeers: 1
* by the → dozen: 1, other: 1, woodman: 1
* hundred and → seventy: 3
* in the → capital: 1, dark: 1, earthly: 1, hand: 1, life: 1, light: 1, midst: 1, rain: 1, rough: 1, superlative: 1, woods: 1
* it was → clearer: 1, the: 11
* of the → captain: 1, chickens: 1, cock: 1, common: 1, failure: 1, heavy: 1, large: 1, law: 1, plain: 1, revolution: 1, shield: 1, state: 1
* on the → mob: 1, musketeers: 1, throne: 2, whole: 1
* one thousand → seven: 3
* seven hundred → and: 3
* seventy five → conduct: 1, environed: 1, spiritual: 1
* there were → a: 2, growing: 1, sheltered: 1
* thousand seven → hundred: 3
* to the → english: 1, human: 1, lords: 1
* was the → age: 2, best: 1, epoch: 2, season: 2, spring: 1, winter: 1, worst: 1, year: 1
* with a → fair: 1, high: 1, large: 2, plain: 1, sack: 1

You can begin to see the flavour of Dickens in this model. For instance, the bigram "was the" shows that, at least in this chapter, Dickens was concerned with time and seasons. 

You should be able to see how we can use this model to generate text. If we're generating text, and we've just generated a particular _n_-gram, we can look up that _n_-gram in the language model and see the words that could come after it. We pick one of the listed words, weighted by the probability of occurrence, and emit that word. That gives us a new _n_-gram, and the process repeats.

For instance, let's say we start with the bigram "it was". We can look up words that come next, and the most most likey is "the". We emit that word and update the "most recent bigram" to be "was the". We pick one words that could follow: "season". Next comes "of", then a choice between "Light" and "Darkness". We can build up more text as we want by repeating the process.


| Emitted text | Current bigram | Word choices | Chosen next word |
|--------------|----------------|--------------|------------------|
| it was | it was | clearer: 1, the: 9 | the |
| it was the | was the | age: 2, best: 1, epoch: 2, season: 2, spring: 1, winter: 1, worst: 1, year: 1 | season |
| it was the season | the season | of: 2 | of |
| it was the season of | season of | Darkness: 1, Light: 1 | Darkness |
| it was the season of Darkness | of Darkness | it: 1 | it |
| it was the season of Darkness it | Darkness it | was: 1 | was |
| it was the season of Darkness it was | it was | clearer: 1, the: 9 | the |
| it was the season of Darkness it was the | was the | age: 2, best: 1, epoch: 2, season: 2, spring: 1, winter: 1, worst: 1, year: 1 | age |
| it was the season of Darkness it was the age | the age | of: 2 | of |
| it was the season of Darkness it was the age of | age of | foolishness: 1, wisdom: 1 | wisdom |
| it was the season of Darkness it was the age of wisdom | of wisdom | it: 1 | it |
| it was the season of Darkness it was the age of wisdom it | wisdom it | was: 1 | was |

Now you have the idea of how the _n_-gram language model is built and used, it's time to implement it. This has three stages.

1. Represent the language model
2. Read some text and populate the model
3. Use the model to generate new text
<!-- #endregion -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Aside: reading text files
<!-- #endregion -->

We need to read large amount of text to populate our language models: a novel's-worth is about the minimum we can get away with. We need to split that text into words (and punctuation). The reading and pre-processing this text is full of fiddly details that aren't worth going into. Instead, we'll just use these couple of functions to do the pre-processing for us.

```python
import re
import string
import collections
import unicodedata
import random
from IPython.display import display, HTML
```

```python jupyter={"outputs_hidden": false}
token_pattern = re.compile(r'[^{}]+'.format(re.escape(string.ascii_letters + string.digits + string.punctuation)))
punctuation_pattern = re.compile('(\d+\.\d+|\w+\'\w+|[{0}]+(?=\w)|(?<=\w)[{0}]+|[{0}]+$)'.format(re.escape(string.punctuation)))
```

```python
def tokenise(text):
    """Split a text string into tokens, splitting on spaces and punctuation,
    but keeping multiple punctuation characters as one token."""
    return [ch for gp in [re.split(punctuation_pattern, t) for t in re.split(token_pattern, text)]
        for ch in gp if ch]
```

```python jupyter={"outputs_hidden": false}
def sjoin(tokens):
    """Combine a set of tokens into a string for pretty-printing."""
    sentence = ''
    for t in tokens:
        if t[-1] not in ".,:;')-!?":
            sentence += ' '
        sentence += t
    return sentence.strip()
```

```python
sample_text = 'The cat sat on the mat. The quick brown fox jumped over the lazy dog.'
tokenise(sample_text)
```

```python
sjoin(tokenise(sample_text))
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Representing the language model
<!-- #endregion -->

Now we understand what the language model should look like, we can work out how to represent it in Python.

The language model is a two-layered data structure. We have a bunch of _n_-grams; for each _n_-gram, we have a bunch of word choices; for each word choice we have a frequency of occurrence. These are key-value stores, so Python `dict`s are the obvious choice. That gives us a structure that looks like this:

**Language model**
| Key | Value |
|-----|-------|
| _n_-gram | word choices |

**Word choices**
| Key | Value |
|-----|-------|
| word | frequency |

However, Python provides a couple of variations on `dict`s, in the [`collections`](https://docs.python.org/3/library/collections.html) library, that will make our lives easier.

The first is a `Counter`, a `dict` specialised for counting things. We'll use this for counting the frequency of words. If we pass a sequence of things to a `Counter`, we get the counts of how often each thing occurs.

```python
counts = collections.Counter(tokenise("the cat sat on the mat"))
counts
```

If we ask about a thing, we're told how often it occurs. Unknown keys don't generate an error, but return a count of zero.

```python
counts['the'], counts['aardvark']
```

If we want to count more things, we use the `update` method and pass in the new things to be counted.

```python
counts.update(['the', 'quick', 'brown'])
counts
```

The other useful `dict` variant is a `defaultdict`. This behaves exactly like a normal `dict` except it gives a default value if we ask for a missing key.

```python
dd = collections.defaultdict(str)
dd[3] = 'hat'
dd[6] = 'banana'
dd
```

```python
dd[3], dd[99]
```

Using a `defaultdict` means we don't have to check an element exists before we update it.

The other wrinkle is that Python won't let us use (mutable) lists of words as the keys to a `dict`-like structure, so we have to convert each _n_-grams from a `list` to a `tuple`.

But with all those implementation details out of the way, let's get on with some programming!

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Buidling the language model
<!-- #endregion -->

With the utilities above, we can read a text file and split it into tokens. Our next job is to use that stream of tokens to build the language model.

We'll build this up in stages, working from finding _n_-grams in a list to building the whole language model.

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Exercise
<!-- #endregion -->

Write a piece of code that will find and print the trigrams (three-word slices) of `tokenise(sample_text)`. The last couple could well be shorter than three words.

```python

```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### Solution
<!-- #endregion -->

```python
sentence = tokenise(sample_text)
for i in range(len(sentence)):
    print(sentence[i:i+3])
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Exercise
<!-- #endregion -->

Modify that code so it doesn't generate the final too-short trigram. This will mean stopping the loop earlier. 

```python

```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### Solution
<!-- #endregion -->

```python
sentence = tokenise(sample_text)
for i in range(len(sentence)-2):
    print(sentence[i:i+3])
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Exercise
<!-- #endregion -->

Modify the code above to find the bigram we want and the next token. Convert the bigram from a list to a tuple and store it in a variable `ngram`. Store the next token in a variable `next_token`. 

```python

```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### Solution
<!-- #endregion -->

```python
sentence = tokenise(sample_text)
for i in range(len(sentence)-2):
    ngram = tuple(sentence[i:i+2])
    next_token = sentence[i+2]
    print(ngram, next_token)
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Exercise
<!-- #endregion -->

<!-- #region -->
Now store these bigrams and next tokens in a language model.

You can create an empty language model with the line

```python
model = collections.defaultdict(collections.Counter)
```

You can push these results into our language model, with the line

```python
model[ngram].update([next_token])
```
<!-- #endregion -->

```python

```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### Solution
<!-- #endregion -->

```python
sentence = tokenise(sample_text)
model = collections.defaultdict(collections.Counter)
for i in range(len(sentence)-2):
    ngram = tuple(sentence[i:i+2])
    next_token = sentence[i+2]
    model[ngram].update([next_token])
model
```

The final step is to take the code we've written and wrap it in a function definition, to make it easy to call for each sentence we process.

While we're at it, we get rid of the "magic number" 2 in the code, and replace it with a parameter for the tuple size.

That gives us the function below.

```python
def build_model(text, tuple_size=2):
    model = collections.defaultdict(collections.Counter)
    # Record each n-gram in turn
    for i in range(len(text) - tuple_size):
        n_gram = text[i:i+tuple_size]
        next_word = text[i+tuple_size]
        model[tuple(n_gram)].update([next_word])
    return model
```

```python jupyter={"outputs_hidden": false}
sample_model = build_model(tokenise(sample_text))
sample_model
```

```python
sample_chapter = open('tale-of-two-cities.txt').read()[1882:7653]
sample_chapter_model = build_model(tokenise(sample_chapter.lower()), tuple_size=2)
len(sample_chapter_model)
```

```python
sample_chapter_model[('it', 'was')]
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Generating text
<!-- #endregion -->

Now we have a model, we can use it to generate text.

The process is roughly the reverse of how we created the language model. We have a current _n_-gram. We find that _n_-gram in the language model, look up the possible next tokens, and pick one. We update the current _n_-gram to include this next token, and the process repeats. Meanwhile, we keep track of all the generated tokens.

A typical generation run is below. You can see how the language model guides the generation of the text.

| Generated text | Current _n_-gram | Next token options | Chosen token |
|----------------|------------------|--------------------|--------------|
| it was | it was | the → 11, clearer → 1 | the |
| it was the | was the | age → 2, epoch → 2, season → 2, best → 1, worst → 1, spring → 1, winter → 1, year → 1 | season |
| it was the season | the season | of → 2 | of |
| it was the season of | season of | light → 1, darkness → 1 | darkness |
| it was the season of darkness | of darkness | , → 1 | , |
| it was the season of darkness , | darkness , | it → 1 | it |
| it was the season of darkness , it | , it | was → 9 | was |
| it was the season of darkness , it was | it was | the → 11, clearer → 1 | the |
| it was the season of darkness , it was the | was the | age → 2, epoch → 2, season → 2, best → 1, worst → 1, spring → 1, winter → 1, year → 1 | age |
| it was the season of darkness , it was the age | the age | of → 2 | of |
| it was the season of darkness , it was the age of | age of | wisdom → 1, foolishness → 1 | foolishness |
| it was the season of darkness , it was the age of foolishness | of foolishness | , → 1 | , |
| it was the season of darkness , it was the age of foolishness , | foolishness , | it → 1 | it |
| it was the season of darkness , it was the age of foolishness , it | , it | was → 9 | was |
| it was the season of darkness , it was the age of foolishness , it was | it was | the → 11, clearer → 1 | the |
| it was the season of darkness , it was the age of foolishness , it was the | was the | age → 2, epoch → 2, season → 2, best → 1, worst → 1, spring → 1, winter → 1, year → 1 | year |
| it was the season of darkness , it was the age of foolishness , it was the year | the year | of → 1, one → 1 | of |
| it was the season of darkness , it was the age of foolishness , it was the year of | year of | our → 1 | our |


<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Picking a random next token
<!-- #endregion -->

One thing we need to do is pick a suitable next token, given a particular _n_-gram and a langauge model.

Python's built-in `random` library has a function `choice()` that will select a random element from a list.

The `Counter` object has a method `elements()` that will return all the items in the `Counter`, each appearing as many times as its count. `elements()` returns an _iterator_, so we need to wrap it in a call to `list` to convert it to the list that `choice` needs.

This cell generates all possible next words for a given _n_-gram:

```python
list(sample_chapter_model[('was', 'the')].elements())
```

This cell picks one of them at random. If you run this cell a few times, you should see different results most of the time.

```python
random.choice(list(sample_chapter_model[('was', 'the')].elements()))
```

(There are other choices for how to select the next item, but we won't go into that here.)

This is the procedure that will generate text for us. The body of it is the `while` loop, that generates a new token while the current _n_-gram exists in the language model, and the generated text isn't longer than the limit.

```python
def generate_text(model, starting_ngram=None, max_length=500):
    if starting_ngram:
        current = starting_ngram
    else:
        current = random.choice(list(model))
    generated = list(current)
    while current in model and len(generated) < max_length:
        next_item = random.choice(list(model[current].elements()))
        # print(generated, ':', current, ':', model[current], ':', next_item)
        generated.append(next_item)
        current = current[1:] + (next_item, )
    return generated
```

We can test this with the `sample_model` created above. This will test the procedure runs without errors, but doesn't produce exciting text.

```python jupyter={"outputs_hidden": false}
sjoin(generate_text(sample_model))
```

Next we load the first chapter of _A Tale of Two Cities". For information, we show how many distinct bigrams are in this chapter.

```python
sample_chapter = open('tale-of-two-cities.txt').read()[1882:7653]
sample_chapter_model = build_model(tokenise(sample_chapter.lower()), tuple_size=2)
len(sample_chapter_model)
```

We can now generate som text, starting with the same opening phrase. We limit the output to 20 tokens, but you can increase it if you want.

```python
sjoin(generate_text(sample_chapter_model, starting_ngram=('it', 'was'), max_length=20))
```

With this seeming to work, let's load the whole book…

```python jupyter={"outputs_hidden": false}
two_cities = open('tale-of-two-cities.txt').read()
two_cities_model = build_model(tokenise(two_cities), tuple_size=3)
len(two_cities_model)
```

…and generate some text. Run this cell several times, and you'll see different text generated each time.

```python
sjoin(generate_text(two_cities_model,  max_length=100))
```

Just printing the text gives annoying breaks in the middle of words. If we produce HTML text, the browser will make it prettier for us.

```python
def pprint(tokens):
    display(HTML(sjoin(tokens)))
```

```python
pprint(generate_text(two_cities_model,  max_length=100))
```

```python jupyter={"outputs_hidden": false}
odyssey = open('odyssey.txt').read()
odyssey_model = build_model(tokenise(odyssey), tuple_size=3)
len(odyssey_model)
```

```python jupyter={"outputs_hidden": false}
pride = open('pride-and-prejudice.txt').read()
pride_model = build_model(tokenise(pride), tuple_size=3)
len(pride_model)
```

```python jupyter={"outputs_hidden": false}
arthur = open('le-mort-d-arthur.txt').read()
arthur_model = build_model(tokenise(arthur), tuple_size=3)
len(arthur_model)
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Generating random text
<!-- #endregion -->

We've got all the parts. Let's generate some large pieces of text. Do these have the same style as the originals? Are those styles distinctive enough to tell which model generated which text?

```python
pprint(generate_text(odyssey_model))
```

```python
pprint(generate_text(two_cities_model))
```

```python
pprint(generate_text(pride_model))
```

```python
pprint(generate_text(arthur_model))
```

```python

```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Merging models
<!-- #endregion -->

We have several language models. As you've seen, each generates text in the style of the source text. What happens if we combine models?

The mechanics of this are fairly easy. The `Counter`s we're using can be added with the `+` operator. This does what you'd expect, and adds all the counts of the two `Counter`s.

```python
c1 = collections.Counter(tokenise("the cat sat on the mat"))
c2 = collections.Counter(tokenise("the cat lay on the bed"))
```

```python
c1, c2, c1 + c2
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Exercise
<!-- #endregion -->

<!-- #region -->
Given the two models below, create a **new** model that combines them. The result should be:

```python
defaultdict(collections.Counter,
            {('the', 'cat'): Counter({'sat': 1, 'lay': 1}),
             ('cat', 'sat'): Counter({'on': 1}),
             ('sat', 'on'): Counter({'the': 1}),
             ('on', 'the'): Counter({'mat': 1, 'bed': 1}),
             ('cat', 'lay'): Counter({'on': 1}),
             ('lay', 'on'): Counter({'the': 1})})
```

(The order of elements in the model may vary, but the contents should be the same.)

Ensure that both source models remain unchanged by the merge.
<!-- #endregion -->

```python
m1 = build_model(tokenise("the cat sat on the mat"))
m2 = build_model(tokenise("the cat lay on the bed"))
m1, m2
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### Solution
<!-- #endregion -->

```python
m12 = collections.defaultdict(collections.Counter)
for k in m1:
    m12[k] += m1[k]
for k in m2:
    m12[k] += m2[k]
m12, m1, m2
```

We can wrap that up in a function.

```python
def merge_models(model1, model2):
    merged = collections.defaultdict(collections.Counter)
    for k in model1:
        merged[k] += model1[k]
    for k in model2:
        merged[k] += model2[k]
    return merged
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### Generating merged-model text
<!-- #endregion -->

Let's combine some models and generate text with them. What does this generated text read like? What style is it in?

```python
two_cities_pride_model = merge_models(two_cities_model, pride_model)
pprint(generate_text(two_cities_pride_model)) 
```

```python
two_cities_odyssey_model = merge_models(two_cities_model, odyssey_model)
pprint(generate_text(two_cities_odyssey_model))
```


```python
arthur_pride_model = merge_models(arthur_model, pride_model)
pprint(generate_text(arthur_pride_model, max_length=500))
```

You'll probably see the text in different styles. The combination of _Pride and Prejudice_ and _A Tale of Two Cities_ isn't too jarring, as the source texts are similar in style. But the combination of a ninteenth-century novel and an ancient epic poem is likely to have a disconcerting effect. You should be able to see a sentence or two that came from one source, then it switches to a sentence or two from the other. 

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Conclusions
<!-- #endregion -->

You've seen a couple of things in this activity. 

First, you should have the idea of what a _language model_ is. It's nothing clever, just some data derived from source text that describes what's "normal" or "expected" from that source. The "large language models" of ChatGPT and similar work in a different way, and are much larger than this, but are doing essentially the same thing.

Second, you've seen how "style" is something that can be encoded in a langauge model quite easily, and replicated fairly well.

Third, you've seen how even randomly generated output can appear somewhat plausible if we don't engage our critical faculties enough to really pay attention to the output we see. Don't believe the hype around the current crop of AI systems!

<!-- #region jp-MarkdownHeadingCollapsed=true -->
# Acknowledgements

All the source texts used here come from [Project Gutenberg](https://www.gutenberg.org/), an online source of public domain works, with [certain permissions and conditions attached](https://www.gutenberg.org/policy/permission.html) . I've modified the books slightly from the versions available there, to remove the legal licence boilerplate and convert some characters to ASCII equivalents. 
<!-- #endregion -->

```python

```
