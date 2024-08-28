# # ELIZA, a simple chatbot

# In this activity, you'll build a version of _ELIZA_, the first chatbot. 
#
# An AI system that has conversations with people is commonly called a chatbot. You could well have encountered them as the first level of customer service for large companies, or the question-answering assistant in phones and smart speakers. Just like the term AI the chatbot has a long history. The first chatbot was ELIZA (Weizenbaum, 1966), which purported to be a psychotherapist. It worked by simple pattern matching on the text inputs it was given.
#
# ELIZA had a set of rules that specified some keywords in the input: if a rule matched the input, then the rule specified how to use it to create a response.
#
# For instance, consider if ELIZA had the rules specified below.
#
# | Pattern | Responses |
# |---------|-----------|
# | ?X I want ?Y | what would it mean if you got ?Y<br>why do you want ?Y<br>suppose you got ?Y soon |
# | ?X I am sad ?Y | I am sorry to hear you are depressed<br>I’m sure it’s not pleasant to be sad |
# | ?X are like ?Y | what resemblance do you see between ?X and ?Y |
#
# With these rules, the sentence 
#
# > sometimes I want to be happy
#
# would match the first rule, with the variable `?Y` matching the fragment ‘to be happy’. 
#
# | ?X | | ?Y |
# |----|-|----|
# |sometimes | I want | to be happy |
#
# ELIZA could use that information to respond with the sentence:
#
# > ‘Why do you want to be happy?’
#
# with the matched value for `?Y` filling out the response.
#
# Weizenbaum created ELIZA to show how text communication could be simulated by such superficial processing. However, he was surprised by how readily people took to ELIZA and engaged it in conversations that were meaningful to the person. ELIZA is simple enough that it’s easy to modify its behaviour, which you will do in this activity.
#
# The state of the art of chatbots has moved on since the 1960s and tools like ChatGPT can now have conversations that seem like they are with a person. While ChatGPT is much more sophisticated than ELIZA, its basic operation remains the same: analyses its input, identifies the salient parts, and uses them to construct the response.

import yaml
import collections
import random
from dataclasses import dataclass


# # Rules

# First, we define a couple of dataclasses to hold things that are important for ELIZA. First is a `Rule`, that consists of a `pattern` and a list of `responses`.

@dataclass
class Rule:
    pattern: list[str]
    responses: list[list[str]]


# We can now read in the set of rules from the file `rules.yaml` (open that file from the list on the left).

def read_rules(rules_file):
    """Read a set of Eliza rules from a file."""
    rules = []
    with open(rules_file) as f:
        for r in yaml.safe_load(f):
            resps = []
            for rsp in r['responses']:
                resps += [rsp.split()]
            rule = Rule(pattern=r['pattern'].split(),
                      responses=resps)
            rules += [rule]
    return rules


all_rules = read_rules('rules.yaml')
all_rules[:3]


# Showing the rules like this is accurate, but hardly easy to read. We'll use this function to "pretty print" a rule into something nicer to look at.

def show_rule(rule):
    """Pretty print a rule."""
    return {'pattern': ' '.join(rule.pattern),
           'responses': [' '.join(resp) for resp in rule.responses]}


for rule in all_rules[5:8]:
    print(show_rule(rule))


# # Matches

@dataclass
class Match:
    text: list[str]
    pattern: list[str]
    bindings: dict[str, list[str]]


# +
# Match = collections.namedtuple('Match', 'text, rule, bindings')
# -

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

halt_words = 'quit halt exit stop'.split()


# +
# def read_rules(rules_file):
#     """Read a set of Eliza rules from a file."""
#     with open(rules_file) as f:
#         rules = [{'pattern': r['pattern'].split(),
#                  'responses': [t.split() for t in r['responses']]}
#             for r in yaml.safe_load(f)]
#     return rules

# +
# def read_rules(rules_file):
#     """Read a set of Eliza rules from a file."""
#     with open(rules_file) as f:
#         rules = [Rule(pattern=r['pattern'].split(),
#                       responses=[t.split() for t in r['responses']])
#             for r in yaml.safe_load(f)]
#     return rules
# -

def is_var(word):
    """True if word is a variable."""
    return word[0] == '?'


def splits(items):
    """Return all the ways of splitting the items into two parts, 
    either of which can be empty."""
    results = []
    for i in range(len(items) + 1):
        results += [(items[:i], items[i:])]
    return results


def match(text, pattern):
    """Match a text against a pattern, returning all possible matches."""
    successes = []
    matches = [Match(text, pattern, {})]
    while matches:
        # print(matches, successes)
        current = matches[0]
        new_matches = []
        if successful_match(current):
            successes += [current.bindings]
        elif current.pattern:
            new_matches = match_item(current.text, current.pattern, current.bindings)
        matches = matches[1:] + new_matches
    return successes


def successful_match(match):
    """A match is successful it it uses all the text and all the pattern."""
    return match.text == [] and match.pattern == []


def match_item(text, pattern, bindings):
    """Match the first segment of text against the first segment of rule,
    respecting the current variable bindings, and creating new bindings
    if necessary."""
    r0 = pattern[0]
    if is_var(r0):
        if r0 in bindings:
            # already seen this variable
            if text[:len(bindings[r0])] == bindings[r0]:
                return [Match(text[(len(bindings[r0])):], paattern[1:], bindings)]
            else:
                return []
        else:
            # not seen this variable yet
            matches = []
            for pre, suf in splits(text):
                new_bindings = bindings.copy()
                new_bindings[r0] = pre
                matches += [Match(suf, pattern[1:], new_bindings)]
            return matches
    elif text and text[0] == r0:
        return [Match(text[1:], pattern[1:], bindings)]
    else:
        return []


def candidate_rules(rules, text):
    """Find all the rules that match the text, returning
    the rules and the bindings that allow the match."""
    return [(rule, bindings) 
            for rule in rules 
            for bindings in match(text, rule.pattern)]


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


def pronoun_person_swap(bindings):
    """Swap pronouns in the given bindings."""
    def swapped(words):
        sw = []
        for w in words:
            if w in pronoun_swaps:
                sw += [pronoun_swaps[w]]
            else:
                sw += [w]
        return sw
    
    return {var: swapped(bindings[var]) for var in bindings}


def respond(rule, bindings):
    """Create a single response from a rule."""
    return fill(random.choice(rule['responses']), bindings)


def eliza_loop(rules):
    """Do the main Eliza loop, asking for input and generating responses."""
    print("Hello. I'm Eliza. What seems to be the problem?")
    while True:
        c = input("> ")
        if c.strip() in halt_words: break
        comment = c.split()
        rule, bindings = candidate_rules(rules, comment)[0]
        swapped_bindings = pronoun_person_swap(bindings)
        print(' '.join(respond(rule, swapped_bindings)))


candidate_rules(all_rules, 'my computer is a vegetable'.split())






