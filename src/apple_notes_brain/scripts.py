"""AppleScript templates used by the Notes MCP server.

Each template uses `__PLACEHOLDER__` markers that `fill()` substitutes after
callers have escaped the values with `applescript.quote` / `applescript.as_list`.
"""
from __future__ import annotations


LIST_FOLDERS = r'''
tell application "Notes"
    set US to (character id 31) as text
    set RS to (character id 30) as text
    set output to ""
    set allFolders to every folder
    repeat with f in allFolders
        set fId to id of f
        set pathStr to my folderPath(f)
        set output to output & fId & US & pathStr & RS
    end repeat
    return output
end tell

on folderPath(f)
    tell application "Notes"
        set parts to {}
        set cur to f
        repeat
            set beginning of parts to name of cur
            try
                set par to container of cur
                if class of par is folder then
                    set cur to par
                else
                    exit repeat
                end if
            on error
                exit repeat
            end try
        end repeat
    end tell
    set AppleScript's text item delimiters to "/"
    set p to parts as text
    set AppleScript's text item delimiters to ""
    return p
end folderPath
'''

LIST_NOTES_ALL = r'''
tell application "Notes"
    set US to (character id 31) as text
    set RS to (character id 30) as text
    set output to ""
    set allNotes to notes
    set cnt to count of allNotes
    if cnt > __LIMIT__ then set cnt to __LIMIT__
    repeat with i from 1 to cnt
        try
            set n to item i of allNotes
            set nId to id of n
            set nName to name of n as text
            try
                set nCid to id of (container of n)
            on error
                set nCid to ""
            end try
            set nMod to (modification date of n) as string
            set output to output & nId & US & nName & US & nCid & US & nMod & RS
        end try
    end repeat
    return output
end tell
'''

LIST_NOTES_SCOPED = r'''
tell application "Notes"
    set US to (character id 31) as text
    set RS to (character id 30) as text
    set output to ""
    set allowedIds to __ALLOWED_IDS__
    set total to 0
    repeat with fid in allowedIds
        if total >= __LIMIT__ then exit repeat
        try
            set fidText to (fid as text)
            set f to first folder whose id is fidText
            set theseNotes to notes of f
            repeat with n in theseNotes
                if total >= __LIMIT__ then exit repeat
                try
                    set nId to id of n
                    set nName to name of n as text
                    set nMod to (modification date of n) as string
                    set output to output & nId & US & nName & US & fidText & US & nMod & RS
                    set total to total + 1
                end try
            end repeat
        end try
    end repeat
    return output
end tell
'''

SEARCH_BODY = r'''
tell application "Notes"
    set US to (character id 31) as text
    set RS to (character id 30) as text
    set output to ""
    set matches to (every note whose (body contains __QUERY__) or (name contains __QUERY__))
    set cnt to count of matches
    if cnt > __LIMIT__ then set cnt to __LIMIT__
    repeat with i from 1 to cnt
        try
            set n to item i of matches
            set nId to id of n
            set nName to name of n as text
            try
                set nCid to id of (container of n)
            on error
                set nCid to ""
            end try
            set nMod to (modification date of n) as string
            set nBody to body of n as text
            set output to output & nId & US & nName & US & nCid & US & nMod & US & nBody & RS
        end try
    end repeat
    return output
end tell
'''

SEARCH_TITLE = r'''
tell application "Notes"
    set US to (character id 31) as text
    set RS to (character id 30) as text
    set output to ""
    set matches to (every note whose name contains __QUERY__)
    set cnt to count of matches
    if cnt > __LIMIT__ then set cnt to __LIMIT__
    repeat with i from 1 to cnt
        try
            set n to item i of matches
            set nId to id of n
            set nName to name of n as text
            try
                set nCid to id of (container of n)
            on error
                set nCid to ""
            end try
            set nMod to (modification date of n) as string
            set output to output & nId & US & nName & US & nCid & US & nMod & RS
        end try
    end repeat
    return output
end tell
'''

GET_NOTE = r'''
tell application "Notes"
    set US to (character id 31) as text
    set n to first note whose id is __NOTE_ID__
    set nId to id of n
    set nName to name of n as text
    try
        set nCid to id of (container of n)
    on error
        set nCid to ""
    end try
    set nMod to (modification date of n) as string
    set nBody to body of n as text
    return nId & US & nName & US & nCid & US & nMod & US & nBody
end tell
'''

CREATE_NOTE_DEFAULT = r'''
tell application "Notes"
    set newNote to make new note with properties {name:__TITLE__, body:__BODY__}
    return id of newNote
end tell
'''

CREATE_NOTE_IN_FOLDER = r'''
tell application "Notes"
    set f to first folder whose id is __FOLDER_ID__
    set newNote to make new note at f with properties {name:__TITLE__, body:__BODY__}
    return id of newNote
end tell
'''

UPDATE_NOTE_REPLACE = r'''
tell application "Notes"
    set n to first note whose id is __NOTE_ID__
    set body of n to __BODY__
    return id of n
end tell
'''

UPDATE_NOTE_APPEND = r'''
tell application "Notes"
    set n to first note whose id is __NOTE_ID__
    set body of n to ((body of n) & __BODY__)
    return id of n
end tell
'''

DELETE_NOTE = r'''
tell application "Notes"
    set n to first note whose id is __NOTE_ID__
    delete n
end tell
'''

RENAME_NOTE = r'''
tell application "Notes"
    set n to first note whose id is __NOTE_ID__
    set name of n to __TITLE__
    return id of n
end tell
'''

MOVE_NOTE = r'''
tell application "Notes"
    set n to first note whose id is __NOTE_ID__
    set f to first folder whose id is __FOLDER_ID__
    move n to f
    return id of n
end tell
'''

# Bulk variant: move N notes to one destination in a single AppleScript invocation.
# Critical for delete_folder cascade — replaces N osascript subprocess calls (one
# per note + ~25 refresh pings each in the verification poll) with ONE subprocess
# call and ONE bulk verification poll. Cuts the bridge-load by ~16x for a 10-note
# cascade and prevents the rapid-fire NSXPCConnection corruption that v10 saw.
#
# Attempts a two-stage strategy inside the SAME tell block:
#   1. FAST PATH: `move (every note whose id is in {...}) to f` — single move
#      command, may let Notes.app coalesce into one CloudKit modification
#      (sub-second commit even for 20+ notes if supported).
#   2. FALLBACK: per-note loop if the fast path errors (e.g. older macOS
#      Notes.app that doesn't support vectorized move). Same correctness as
#      the original implementation.
# Either way: ONE osascript subprocess, ONE tell block, ALL moves sequenced.
BULK_MOVE_NOTES = r'''
tell application "Notes"
    set f to first folder whose id is __FOLDER_ID__
    set noteIds to __NOTE_IDS__
    set okCount to 0
    set failCount to 0
    try
        -- Fast path: single batch move command.
        move (every note whose id is in noteIds) to f
        set okCount to (count of noteIds)
    on error
        -- Fallback: per-note loop inside the same tell block.
        repeat with nid in noteIds
            try
                set n to first note whose id is (nid as text)
                move n to f
                set okCount to okCount + 1
            on error
                set failCount to failCount + 1
            end try
        end repeat
    end try
    return (okCount as text) & "/" & (failCount as text)
end tell
'''

# Bulk variant: create N notes in one folder in a single AppleScript invocation.
# Replaces N osascript subprocess calls with one. Each note still triggers its own
# CoreData/CloudKit save inside Notes.app (Apple Cocoa Scripting has no batch
# entry point — confirmed via the .sdef research), so the per-note save lag is
# unchanged. The win is purely subprocess startup cost: ~50ms × N saved.
#
# Title and body lists are kept in sync — index i in __TITLES__ matches index i
# in __BODIES__. Empty body emits an empty note. Output is RECORD_SEP-joined
# AppleScript note ids (the long URIs); caller parses to short ids via db.short_id.
BULK_CREATE_NOTES_IN_FOLDER = r'''
tell application "Notes"
    set RS to (character id 30) as text
    set f to first folder whose id is __FOLDER_ID__
    set titles to __TITLES__
    set bodies to __BODIES__
    set output to ""
    set n to count of titles
    repeat with i from 1 to n
        try
            set t to item i of titles
            set b to item i of bodies
            set newNote to make new note at f with properties {name:t, body:b}
            set output to output & (id of newNote) & RS
        end try
    end repeat
    return output
end tell
'''

# Same shape but creates notes in the user's default folder (no folder-arg lookup).
BULK_CREATE_NOTES_DEFAULT = r'''
tell application "Notes"
    set RS to (character id 30) as text
    set titles to __TITLES__
    set bodies to __BODIES__
    set output to ""
    set n to count of titles
    repeat with i from 1 to n
        try
            set t to item i of titles
            set b to item i of bodies
            set newNote to make new note with properties {name:t, body:b}
            set output to output & (id of newNote) & RS
        end try
    end repeat
    return output
end tell
'''


CREATE_FOLDER_DEFAULT = r'''
tell application "Notes"
    set newFolder to make new folder with properties {name:__NAME__}
    return id of newFolder
end tell
'''

CREATE_FOLDER_IN_FOLDER = r'''
tell application "Notes"
    set parentFolder to first folder whose id is __PARENT_ID__
    set newFolder to make new folder at parentFolder with properties {name:__NAME__}
    return id of newFolder
end tell
'''


RENAME_FOLDER = r'''
tell application "Notes"
    set f to first folder whose id is __FOLDER_ID__
    set name of f to __NAME__
    return id of f
end tell
'''

DELETE_FOLDER = r'''
tell application "Notes"
    set f to first folder whose id is __FOLDER_ID__
    delete f
end tell
'''


def fill(template: str, **kwargs: str) -> str:
    """Replace __KEY__ markers with values. Values must already be AppleScript-safe."""
    out = template
    for key, val in kwargs.items():
        out = out.replace(f"__{key}__", val)
    return out
