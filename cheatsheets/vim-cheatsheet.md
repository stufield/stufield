# VIM Cheatsheet

Stu Field

19 September 2024

--------------------------------

# Movement

| key      | action      |
|:-------- |:----------- |
| `h`      | move left   |
| `j`      | move down   |
| `k`      | move up     |
| `l`      | move right  |
| `w`      | jump by start of words (punctuation considered words) |
| `W`      | jump by words (spaces separate words)   |
| `e`      | jump to end of words (punctuation considered words)   |
| `E`      | jump to end of words (no punctuation)   |
| `b`      | jump backward by words (punctuation considered words) |
| `B`      | jump backward by words (no punctuation) |
| `0`      | (zero) start of line                    |
| `ctrl+u` | page up     |
| `ctrl+d` | page down   |
| `^`      | first non-blank character of line       |
| `$`      | end of line |
| `G`      | Go To command (prefix with number -> 5G goes to line 5) |


# Editing


| key     | action      |
|:------- |:----------- |
| `r`     | replace a single character (does not use insert mode) |
| `J`     | join line below to the current one |
| `cc`    | change (replace) an entire line |
| `cw`    | change (replace) to the end of word |
| `c$`    | change (replace) to the end of line |
| `s`     | delete character at cursor and substitute text |
| `S`     | delete line at cursor and substitute text (same as cc) |
| `xp`    | transpose two letters (delete and paste, technically) |
| `u`     | undo |
| `.`     | repeat last command |


# Insert Mode

| key     | action      |
|:------- |:----------- |
| `i`     | start insert mode at cursor         |
| `I`     | insert at the beginning of the line |
| `a`     | append after the cursor             |
| `A`     | append at the end of the line       |
| `o`     | open (append) blank line below current line (no need to press return) |
| `O`     | open blank line above current line  |
| `ea`    | append at end of word               |
| `Esc`   | exit insert mode                    |


# Visual Mode

| key      | action      |
|:-------- |:----------- |
| `v`      | start visual mode, mark lines,   |
|          | then do command (such as y-yank) |
| `V`      | start Line-wise visual mode      |
| `o`      | move to other end of marked area |
| `ctrl+v` | start visual block mode          |
| `O`      | move to Other corner of block    |
| `aw`     | mark a word                      |
| `ab`     | a () block (with braces)         |
| `aB`     | a `\` block (with brackets)      |
| `ib`     | inner () block                   |
| `iB`     | inner `\` block                  |
| `Esc`    | exit visual mode                 |


# Visual Commands

| key    | action      |
|:------ |:----------- |
| `>`    | shift right |
| `<`    | shift left  |
| `y`    | yank (copy) marked text |
| `d`    | delete marked text      |
| `~`    | switch case |


# Cut & Paste

| key    | action      |
|:------ |:----------- |
| `yy`   | yank (copy) a line |
| `yl`   | yank character under cursor |
| `2yy`  | yank 2 lines |
| `yw`   | yank word |
| `y$`   | yank to end of line |
| `p`    | put (paste) the clipboard after cursor |
| `P`    | put (paste) before cursor |
| `dd`   | delete (cut) a line |
| `dw`   | delete (cut) the current word |
| `x`    | delete (cut) current character |

# Exiting

| key    | action      |
|:------ |:----------- |
| `:w`   | write (save) the file, but don't exit |
| `:wq`  | write (save) and quit |
| `:q`   | quit (fails if anything has changed) |
| `:q!`  | quit and throw away changes |

# Search & Replace

| key               | action      |
|:----------------- |:----------- |
| `/pattern`        | search for pattern |
| `?pattern`        | search backward for pattern |
| `n`               | repeat search in same direction |
| `N`               | repeat search in opposite direction |
| `:\%s/old/new/g`  | replace all _old_ with _new_ throughout file |
| `:\%s/old/new/gc` | replace all _old_ with _new_ throughout file with confirmations |

# Multiple Files

| key               | action      |
|:----------------- |:----------- |
| `:e filename`     | Edit a file in a new buffer |
| `:bnext (or :bn)` | go to next buffer |
| `:bprev (of :bp)` | go to previous buffer |
| `:bd`             | delete a buffer (close a file) |
| `:sp filename`    | Open a file in a new buffer and split window |
| `ctrl+ws`         | Split windows |
| `ctrl+ww`         | switch between windows |
| `ctrl+wq`         | Quit a window |
| `ctrl+wv`         | Split windows vertically |

# VI Spell Check

| key              | action      |
|:---------------- |:----------- |
| `:set spell`     | Turn on spell checking |
| `]s`             | Go to next misspelled word |
| `[s`             | Go to previous misspelled word |
| `]S`             | like `]s` but only stop at bad words not rare words |
| `zg`             | Add word as a good word and append to `spellfile` |
| `zG`             | Add word to internal spell list |
| `zug`            | Remove word from `spellfile` |
| `zuG`            | Remove word from internal spell list |
| `z=`             | List suggestions for misspelled word |
| `:set spell spelllang=en\_us` | Set region dictionary (us, ca, gb, de\_de, de\_at, de\_ch) |
| `:set nospell`   | Turn off spell checking |

