LDK: A seismic array detection algorithm
===============================================

[Project Website](http://dkilb8.wix.com/earthscope)


- [Intro](#intro)
- [Dependencies](#installation)
    - [Pandas](#PANDAS)
    - [Scipy](#SCIPY)
    - [Numpy](#NUMPY)
    - [DETEX](#DETEX)
    
- [Quick Feature Summary](#quick-feature-summary)
- [User Guide](#user-guide)
    - [General Usage](#general-usage)
    
- [Commands](#commands)
    - [YcmCompleter subcommands](#ycmcompleter-subcommands)
- [Options](#options)
- [FAQ](#faq)
- [Contact](#contact)
- [License](#license)

Intro
-----

text here



Installation
------------

### PANDAS

http://pandas.pydata.org/

    pip install pandas



### SCIPY

http://www.scipy.org/


    pip install scipy

### NUMPY


    pip install numpy

### DETEX

https://github.com/dchambers/detex.git


Quick Feature Summary
-----

### General (all languages)

* Super-fast identifier completer including tags files and syntax elements
* Intelligent suggestion ranking and filtering
* File and path suggestions
* Suggestions from Vim's OmniFunc
* UltiSnips snippet suggestions



User Guide
----------

### General Usage

- If the offered completions are too broad, keep typing characters; YCM will
  continue refining the offered completions based on your input.
- Filtering is "smart-case" sensitive; if you are typing only lowercase letters,
  then it's case-insensitive. If your input contains uppercase letters, then the
  uppercase letters in your query must match uppercase letters in the completion
  strings (the lowercase letters still match both). So, "foo" matches "Foo" and
  "foo", "Foo" matches "Foo" and "FOO" but not "foo".
- Use the TAB key to accept a completion and continue pressing TAB to cycle
  through the completions. Use Shift-TAB to cycle backwards. Note that if you're
  using console Vim (that is, not Gvim or MacVim) then it's likely that the
  Shift-TAB binding will not work because the console will not pass it to Vim.
  You can remap the keys; see the _[Options][]_ section below.

Knowing a little bit about how YCM works internally will prevent confusion. YCM
has several completion engines: an identifier-based completer that collects all
of the identifiers in the current file and other files you visit (and your tags
files) and searches them when you type (identifiers are put into per-filetype
groups).


