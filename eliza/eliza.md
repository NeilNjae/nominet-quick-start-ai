# ELIZA, a simple chatbot


In this activity, you'll build a version of _ELIZA_, the first chatbot. 

An AI system that has conversations with people is commonly called a chatbot. You could well have encountered them as the first level of customer service for large companies, or the question-answering assistant in phones and smart speakers. Just like the term AI the chatbot has a long history. The first chatbot was ELIZA (Weizenbaum, 1966), which purported to be a psychotherapist. It worked by simple pattern matching on the text inputs it was given.

ELIZA had a set of rules that specified some keywords in the input: if a rule matched the input, then the rule specified how to use it to create a response.

For instance, consider if ELIZA had the rules specified below.

| Pattern | Responses |
|---------|-----------|
| ?X I want ?Y | what would it mean if you got ?Y<br>why do you want ?Y<br>suppose you got ?Y soon |
| ?X I am sad ?Y | I am sorry to hear you are depressed<br>I’m sure it’s not pleasant to be sad |
| ?X are like ?Y | what resemblance do you see between ?X and ?Y |

With these rules, the sentence 

> sometimes I want to be happy

would match the first rule, with the variable `?Y` matching the fragment ‘to be happy’. 

| ?X | | ?Y |
|----|-|----|
|sometimes | I want | to be happy |

ELIZA could use that information to respond with the sentence:

> ‘Why do you want to be happy?’

with the matched value for `?Y` filling out the response.

Weizenbaum created ELIZA to show how text communication could be simulated by such superficial processing. However, he was surprised by how readily people took to ELIZA and engaged it in conversations that were meaningful to the person. ELIZA is simple enough that it’s easy to modify its behaviour, which you will do in this activity.

The state of the art of chatbots has moved on since the 1960s and tools like ChatGPT can now have conversations that seem like they are with a person. While ChatGPT is much more sophisticated than ELIZA, its basic operation remains the same: analyses its input, identifies the salient parts, and uses them to construct the response.

```python
import yaml
import collections
import random
import string
```

# Building ELIZA


How do we go about building ELIZA?

There are three main components we need to address.
1. Representing the rules
2. Matching user input to rules
3. Using the matches to create ELIZA's output

Once we have these pieces, we can do the fourth stage of putting them all together into the full chatbot.

We'll take each of these steps in turn.


# Rules


We can download some rules to work with. Execute this cell.

```python
!wget --no-check-certificate https://raw.githubusercontent.com/NeilNjae/nominet-quick-start-ai/main/eliza/rules.yaml
```

You can look at the raw rules file by clicking on the "Files" icon in the left sidebar and choosing the `rules.yaml` entry. You should be alboe to see that the rules follow the structure of a pattern and a set of possible responses.


We will follow that structure in our implementation and represent each rule as a dictionary of two elements: a `pattern` and a list of `responses`.


We can now read in the set of rules from the file `rules.yaml`.

```python
def read_rules(rules_file):
    """Read a set of Eliza rules from a file."""
    rules = []
    with open(rules_file) as f:
        for r in yaml.safe_load(f):
            resps = []
            for rsp in r['responses']:
                resps += [rsp.split()]
            rule = {'pattern': r['pattern'].split(),
                    'responses': resps}
            rules += [rule]
    return rules
```

```python
all_rules = read_rules('rules.yaml')
all_rules[:3]
```

Showing the rules like this is accurate, but hardly easy to read. We'll use this function to "pretty print" a rule into something nicer to look at.

```python
def show_rule(rule):
    """Pretty print a rule."""
    return {'pattern': ' '.join(rule['pattern']),
           'responses': [' '.join(resp) for resp in rule['responses']]}
```

```python
for rule in all_rules[5:8]:
    print(show_rule(rule))
```

# Matches


We need to match user input to the rules. The _result_ of this process is for us to know which variables in a rule match which parts of the input; that information can be used to generate the output. The normal term for "connecting a variable and its value" is a _binding_. In the example above, we would say that the variable `?X` is _bound_ to the value "sometimes" and the variable `?Y` is _bound_ to the value "to be happy". We can store a binding as a Python `dict` going from variable name to its bound value, such as `{'?X': ['sometimes'], '?Y': ['to' 'be' 'happy']}`.

We'll do that by walking along the input text and the rule pattern in parallel, from left to right. If we can match the first token of each, we'll record any bindings then move on to the next tokens.

Consider the input text
> the cat sat on the mat

and the rule pattern
> the ?X sat on the ?Y

We start by comparing the leftmost tokens. The leftmost token in in the pattern is the literal word "the" and this is the same as the first word of the input. These are the same, so the matching continues.

We now have the situation of matching "cat sat on the mat" with "?X sat on the ?Y". We need to bind the variable `?X` with some words in the input. The variable can match zero or more words; as yet, just by looking at the first word of the input, we don't know how many words to include in the binding. We'll do what's easy for us and try them all, letting the computer do the work. This gives us a set of possible bindings and next steps:

| Remaining text | Remaining pattern | Bindings |
|----------------|-------------------|----------|
| cat sat on the mat | sat on the ?Y | {'?X': []} |
| sat on the mat | sat on the ?Y | {'?X': ['cat']} |
| on the mat | sat on the ?Y | {'?X': ['cat', 'sat']} |
| the mat | sat on the ?Y | {'?X': ['cat', 'sat', 'on']} |
| mat | sat on the ?Y | {'?X': ['cat', 'sat', 'on', 'the']} |

It's obvious to us that only one of these will lead to the whole pattern matching, but that's all we can do by looking at just the _first_ elements of the text and the pattern.

There's more to look at with how matches work, but this is enough to get along with at the moment.

The first thing to do is define a structure to hold a partial match: we need to hold the remaining text, the remaining pattern, and the bindings found so far. We'll use another `dict` with these three elements.


We define a match as _successful_ if there is both no more text and no more pattern.

```python
def successful_match(match):
    """A match is successful it it uses all the text and all the pattern."""
    return match['text'] == [] and match['pattern'] == []
```

We also have a utility function that will convert the user's input into a list of words. It removes punctuation and converts all the text to lower-case.

```python
def tokenise(text):
    """Convert a text string into a list of lower-case words, removing punctuation"""
    without_punctuation = text.translate(str.maketrans('', '', string.punctuation))
    lowered = without_punctuation.lower()
    return lowered.split()
```

```python
tokenise("A man, a plan, a canal, Panama!")
```

## Exercise


We can now build a couple of utiilty functions to make this all work. 

First is a function that will detect if a token in a pattern is a variable. We mark variables as any string that starts with a "?".

Write the function `is_var()`. Recall that you can find the specific characters of a string by using indexes.

```python
def is_var(word):
    ....
```

### Solution

```python
def is_var(word):
    """True if word is a variable."""
    return word[0] == '?'
```

### End of solution


## Exercise

<!-- #region -->
The next thing is to find candiates for a variable to match. We need to split the text into two parts (a prefix and a suffix); the prefix can be bound to a variable and the suffix used to continue matching.

Write the function `splits()` that returns all two-part splits of a list. Either part can be empty.

For instance, the function call 
```python
splits(['a', 'man', 'a', 'plan'])
```

should return

```python
[([],                        ['a', 'man', 'a', 'plan']),
 (['a'],                     ['man', 'a', 'plan']),
 (['a', 'man'],              ['a', 'plan']),
 (['a', 'man', 'a'],         ['plan']),
 (['a', 'man', 'a', 'plan'], [])]
```

Recall how to find slices of a list.
<!-- #endregion -->

### Solution

```python
def splits(items):
    """Return all the ways of splitting the items into two parts, 
    either of which can be empty."""
    results = []
    for i in range(len(items) + 1):
        results += [(items[:i], items[i:])]
    return results
```

```python
splits(['a', 'man', 'a', 'plan'])
```

### End of solution


We can now match the start of a pattern against the start of some text. This function is given some text, a pattern, and some bindings. It returns a list of `Match` objects that can be used to continue matching from this point. If the given pattern can't match against the given text, the function returns an empty list.

There are a few cases to consider, and the implementation of the function follows them.

1. If the first element of the pattern is a varialble:
   1. If this is a variable that is already bound:
      1. If the first few tokens of the text match the bound value:
         1. Create a new `Match` object that drops the first few tokens from the text, variable from the pattern, and the existing bindings; return it as a singleton list.
      2. Otherwise:
         1. Return this empty list.
    2. Otherwise (this is the first time seeing this variable):
       1. Initialise the list of matches to the empty list.
       2. For each (prefix, suffix) pair from splitting the text:
          1. Bind the variable to the prefix.
          2. Create a new `Match` object, where the text to match is the suffix and the pattern doesn't have this variable.
          3. Add the new `Match` to the list of matches
        3. Return the list of matches
2. Otherwise (the first element of the pattern is a literal word):
   1. If the first element of the pattern is the same as the first element of the text:
      1. Return a new `Match` object that drops this word.
   2. Otherwise:
      1. Return the empty list.
    

```python
def match_item(match):
    """Match the first segment of text against the first segment of rule,
    respecting the current variable bindings, and creating new bindings
    if necessary."""
    text = match['text']
    pattern = match['pattern']
    bindings = match['bindings']
    element = pattern[0]
    if is_var(element):
        if element in bindings:
            # already seen this variable
            if text[:len(bindings[element])] == bindings[element]:
                return [{'text': text[(len(bindings[element])):], 
                         'pattern': pattern[1:], 
                         'bindings': bindings}]
            else:
                return []
        else:
            # not seen this variable yet
            matches = []
            for prefix, suffix in splits(text):
                new_bindings = bindings.copy()
                new_bindings[element] = prefix
                matches += [{'text': suffix, 'pattern': pattern[1:], 'bindings': new_bindings}]
            return matches
    elif text and text[0] == element:
        return [{'text': text[1:], 'pattern': pattern[1:], 'bindings': bindings}]
    else:
        return []
```

```python
match_item({'text': tokenise('a man a plan a canal panama'), 'pattern': ['?X', 'a', '?Y'], 'bindings': {}})
```

```python
match_item({'text': tokenise('a man a plan a canal panama'), 'pattern': ['a', '?Y'], 'bindings': {}})
```

```python
match_item({'text': tokenise('a man a plan a canal panama'), 'pattern': ['?X', 'a', '?Y'], 'bindings': {'?X': ['a', 'man']}})
```

```python
match_item({'text': tokenise('a man a plan a canal panama'), 'pattern': ['?X', 'a', '?Y'], 'bindings': {'?X': ['a', 'plan']}})
```

```python
match_item({'text': tokenise('a man a plan a canal panama'), 'pattern': ['a', 'plan'], 'bindings': {'?X': ['a', 'man']}})
```

```python
match_item({'text': tokenise('a man a plan a canal panama'), 'pattern': ['plan'], 'bindings': {'?X': ['a', 'man']}})
```

Now we can match one item in a pattern, we can write a wrapper that will find all matches between a given text and a given pattern.

We do that by maintianing two lists. One is a list of partial matches in progress, the other is a list of successful matches.

The "matches in progress" starts with just one item: the initial text, the initial item, and the empty set of bindings. The successes starts empty.

We treat the matches in progress as a queue of work. While there is work to do, we take the first item of the queue, find all the results of `match_item` on that partial match, and insert the results at the end of the queue.

The full pseudocode for `match` is:

```
initialise successes to []
initialise matches to [{text, pattern, {} }]
while matches:
  set current to first element of matches
  set new_matches to []
  if current is a successful match:
    add current's bindings to successes
    set new_matches to []
  else if current's pattern is not empty:
    set new_matches to match_item(current)
  set matches to (matches without the first element) + new_matches
return successes
```


## Exericse


Write `match`.

The call to 
```
match(tokenise('a man a plan a canal panama'), ['?X', 'a', '?Y', 'a', '?Z'])
```

should give this result of three possible matches.

```
[{'?X': [], '?Y': ['man'], '?Z': ['plan', 'a', 'canal', 'panama']},
 {'?X': [], '?Y': ['man', 'a', 'plan'], '?Z': ['canal', 'panama']},
 {'?X': ['a', 'man'], '?Y': ['plan'], '?Z': ['canal', 'panama']}]
```


### Solution

```python
def match(text, pattern):
    """Match a text against a pattern, returning all possible matches."""
    successes = []
    matches = [{'text': text, 'pattern': pattern, 'bindings': {}}]
    while matches:
        # print(matches, '***', successes)
        current = matches[0]
        new_matches = []
        if successful_match(current):
            successes += [current['bindings']]
            new_matches = []
        elif current['pattern']:
            new_matches = match_item(current)
        matches = matches[1:] + new_matches
    return successes
```

```python
match(tokenise('a man a plan a canal panama'), ['?X', 'a', '?Y', 'a', '?Z'])
```

### End of solution


There are a couple of complexities here. 

It could be that a pattern matches some text in more than one way. For instance, the pattern `?X a ?Y` has several possible matches to the sentence "a man, a plan, a canal, Panama!"

| ?X | ?Y |
|----|----|
| Nothing | man, a plan, a canal, Panama! |
| a man, | plan, a canal, Panama! |
| a man, a plan, | canal, Panama! |

In this case, we should find all the matches.


# Responses


Now we can _match_ a rule against an input text, it's time to _respond_ to that text.

We'll use the `response` part of the rule, together with the `bindings` we've found, to build the response.

We'll walk along the `response`, element by element. We'll build up the `filled_response` as we go. If the current element of the `response` is a variable, we add the variable's bound value to the `filled_response`; otherwise, the element is a word, and we add it.

For instance, if we have the rule

```
pattern: ?X sat on ?Y
response: does ?Y like ?X ?
```

and the binding `{'?X': ['the', 'cat'], ?Y': ['the', 'mat']}` , we would want to build the response `['does', 'the', 'mat', 'like', 'the', 'cat', '?']`


## Exercise


Write `fill`, that takes a rule response and some bindings, and fills out the response. Note that the `response` is a list of tokens, either words or variables.


### Solution

```python
def fill(response, bindings):
    """Complete a rule's response by using the bindings to fill
    the variable slots."""
    filled_response = []
    for w in response:
        if is_var(w):
            if w in bindings:
                filled_response += bindings[w]
            else:
                filled_response += ['MISSING']
        else:
            filled_response += [w]
    return filled_response
```

### End of solution


# Interaction loop


Now we've got all the parts of ELIZA, it's time to put them together.

The basic interaction loop is:

1. Ask for input
2. Find a matching rule
3. Build the response
4. Repeat forever

Expressed in Python, it looks like this:

```python
def eliza_loop(rules):
    """Do the main Eliza loop, asking for input and generating responses."""
    print("Hello. I'm Eliza. What seems to be the problem?")
    while True:
        c = input("> ")
        comment = tokenise(c)
        if comment[0] in halt_words: break
        rule, bindings = all_matches(rules, comment)[0]
        swapped_bindings = pronoun_person_swap(bindings)
        response = respond(rule, swapped_bindings)
        print(' '.join(response))
```

There are a few loose ends to tidy up.


## Stopping


At some point, we want to stop ELIZA. We do that by having a list of words to halt on. If the first token of the user's input is one of these, we stop.

```python
halt_words = 'quit halt exit stop'.split()
```

## Picking the one rule and binding


Our `match` procedure can return many bindings for a rule, and there could be many rules that match as given input. Which do we choose?

We want to choose the first rule the matches, as that allows us to give specific rules before more general ones. If there are several possible bindings, we could pick any; for convenience, we'll pick the first.

We'll do that by writing a function `all_matches` that returns a list of all rule/binding combinations that match the input. We can then take the first of these in the ELIZA main loop (above).

```python
def all_matches(rules, text):
    """Find all the rules that match the text, returning
    the rules and the bindings that allow the match."""
    results = []
    for rule in rules:
        for bindings in match(text, rule['pattern']):
            results += [(rule, bindings)]
    return results
```

## Pronoun swapping


Consider the rule

* Pattern: 'I like ?X'
* Response: 'Tell me more about ?X'

applied to the sentence "I like my cat".

This would bind with `{'?X': 'my cat'}`, and would in turn generate the response "Tell me more about my cat". This isn't what we want. Instead, we want to swap pronouns in the bindings from first-person to second-person, so the binding would become `{'?X': 'your cat'}` and the response "Tell me more about your cat".

We do this with a somple lookup table (implemented as a `dict`) that maps an original word to its replacement.

```python
pronoun_swaps = {
    'i': 'you',
    'me': 'you',
    'my': 'your',
    'mine': 'yours',
    'am': 'are',
    'you': 'i',
    'your': 'my',
    'yours': 'mine'
}
```

## Exercise


Write the function `pronoun_person_swap` that uses `pronoun_swaps` to change the pronouns in a collection of bindings. You'll need two nested loops (one to cycle through each binding, one to cycle through each word in that binding). For each binding, build up a new bound value, word by word.


### Solution

```python
def pronoun_person_swap(bindings):
    """Swap pronouns in the given bindings."""
    result = {}
    for var in bindings:
        swapped_value = []
        for word in bindings[var]:
            if word in pronoun_swaps:
                swapped_value += [pronoun_swaps[word]]
            else:
                swapped_value += [word]
        result[var] = swapped_value
    return result
```

### End of solution


The final part of the interaction loop is to pick a random response from the ones available. We do that with Python's built-in `random` library.

```python
def respond(rule, bindings):
    """Create a single response from a rule."""
    return fill(random.choice(rule['responses']), bindings)
```

# Using ELIZA


With ELIZA built, it's time to use it!

We call the main loop, using the rules we loaded earlier. Use one of the halt words to stop the interaction.

```python
eliza_loop(all_rules)
```

# Modifying ELIZA


With the basic functionality of ELIZA, we can change its behaviour by changing the rules. 

Open the `rules.yaml` file (use the "Files" icon on the left). You can modify the extisting rules, or add new ones, by typing the new rules into the file. 

Change some of the rules to change how ELIZA interacts. Remember that ELIZA will respond with the first rule that matches, so you're more likely to see your rules in action if you put them near the top of the `rules.yaml` file.

> Note that the formatting must be precise, with the left indents being exactly two spaces. Use an existing rule as a template.

Save the file when you're done. Reload the rules then run ELIZA again.

```python
all_rules = read_rules('rules.yaml')
eliza_loop(all_rules)
```

```python

```
