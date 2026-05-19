-- Minimal NoteStore.sqlite schema for integration tests.
-- Mirrors the columns + tables used by sqlite_reader.py.
-- All columns nullable to match actual NoteStore.sqlite shape.

CREATE TABLE Z_METADATA (
    Z_VERSION INTEGER PRIMARY KEY,
    Z_UUID VARCHAR,
    Z_PLIST BLOB
);
INSERT INTO Z_METADATA (Z_VERSION, Z_UUID) VALUES (1, 'TEST-UUID-12345-67890');

CREATE TABLE Z_PRIMARYKEY (
    Z_ENT INTEGER PRIMARY KEY,
    Z_NAME VARCHAR,
    Z_SUPER INTEGER,
    Z_MAX INTEGER
);
INSERT INTO Z_PRIMARYKEY (Z_ENT, Z_NAME) VALUES (12, 'ICNote'), (15, 'ICFolder'), (14, 'ICAccount');

-- Main syncing-object table. Holds note rows (Z_ENT=12), folder rows (Z_ENT=15),
-- account rows (Z_ENT=14), and attachment rows (Z_ENT=5).
-- ZACCOUNT8 is the latest macOS column name for the account FK on folder rows.
CREATE TABLE ZICCLOUDSYNCINGOBJECT (
    Z_PK INTEGER PRIMARY KEY,
    Z_ENT INTEGER,
    ZTITLE1 TEXT,                  -- note title
    ZTITLE2 TEXT,                  -- folder title
    ZNAME TEXT,                    -- account display name (Z_ENT=14 rows)
    ZFOLDER INTEGER,               -- ZICNote → ZICFolder FK
    ZPARENT INTEGER,               -- ZICFolder → ZICFolder parent FK
    ZNOTE INTEGER,                 -- ZICAttachment → ZICNote FK
    ZNOTEDATA INTEGER,             -- ZICNote → ZICNoteData FK
    ZIDENTIFIER TEXT,
    ZMARKEDFORDELETION INTEGER DEFAULT 0,
    ZFOLDERTYPE INTEGER,           -- 0=normal, 1=trash
    ZMODIFICATIONDATE1 REAL,
    ZCREATIONDATE1 REAL,
    ZACCOUNT8 INTEGER,             -- account FK
    ZISPASSWORDPROTECTED INTEGER DEFAULT 0,
    ZSERVERSHAREDATA BLOB,
    ZZONEOWNERNAME TEXT,
    ZSHARETARGETMANAGEDOBJECTID INTEGER,
    ZISPINNED INTEGER DEFAULT 0,
    ZSNIPPET TEXT,
    ZNEEDSINITIALFETCHFROMCLOUD INTEGER DEFAULT 0,
    ZSERVERRECORDDATA BLOB,
    ZMERGEABLEDATA1 BLOB,
    ZCRYPTOINITIALIZATIONVECTOR BLOB,
    ZLOCKEDNOTESMODE INTEGER,
    -- Attachment metadata (Z_ENT=5 rows). NULL on note/folder/account rows.
    ZTYPEUTI TEXT,
    ZFILENAME TEXT,
    ZFALLBACKPDFGENERATION BLOB    -- non-null marks a scanned-doc attachment
);

-- Account row (Z_ENT=14). ZNAME is the display name used by _account_name_column.
INSERT INTO ZICCLOUDSYNCINGOBJECT (Z_PK, Z_ENT, ZNAME, ZIDENTIFIER, ZSERVERRECORDDATA)
VALUES (1, 14, 'iCloud', 'icloud-account-id', X'01');

-- Folders (Z_ENT=15). All have ZSERVERRECORDDATA non-null so ghost filter
-- doesn't hide them.
INSERT INTO ZICCLOUDSYNCINGOBJECT (Z_PK, Z_ENT, ZTITLE2, ZIDENTIFIER, ZFOLDERTYPE, ZACCOUNT8, ZSERVERRECORDDATA) VALUES
    (2, 15, 'Notes', 'DefaultFolder-CloudKit', 0, 1, X'01'),
    (3, 15, 'Recently Deleted', 'TrashFolder-CloudKit', 1, 1, X'01'),
    (4, 15, 'Work', 'work-folder', 0, 1, X'01'),
    (5, 15, 'Personal', 'personal-folder', 0, 1, X'01'),
    (6, 15, 'Subfolder', 'sub-folder', 0, 1, X'01');
UPDATE ZICCLOUDSYNCINGOBJECT SET ZPARENT = 4 WHERE Z_PK = 6;

-- Notes (Z_ENT=12)
INSERT INTO ZICCLOUDSYNCINGOBJECT (Z_PK, Z_ENT, ZTITLE1, ZIDENTIFIER, ZFOLDER, ZMODIFICATIONDATE1) VALUES
    (10, 12, 'Note in Notes', 'note-1', 2, 7700000),
    (11, 12, 'Note in Work', 'note-2', 4, 7710000),
    (12, 12, 'Note in Subfolder', 'note-3', 6, 7720000),
    (13, 12, 'Trashed Note', 'note-4', 3, 7730000),
    (14, 12, 'Locked Note', 'note-5', 2, 7740000);
UPDATE ZICCLOUDSYNCINGOBJECT SET ZISPASSWORDPROTECTED = 1 WHERE Z_PK = 14;
UPDATE ZICCLOUDSYNCINGOBJECT SET ZMARKEDFORDELETION = 1 WHERE Z_PK = 13;

-- Notes data table (legacy fallback for protobuf bodies)
CREATE TABLE ZICNOTEDATA (
    Z_PK INTEGER PRIMARY KEY,
    Z_ENT INTEGER,
    ZNOTE INTEGER,
    ZDATA BLOB
);

-- ACHANGE table (NSPersistentHistory). Source uses ZENTITYPK (not ZCHANGEDOBJECTID).
CREATE TABLE ACHANGE (
    Z_PK INTEGER PRIMARY KEY,
    ZENTITY INTEGER,
    ZCHANGETYPE INTEGER,           -- 2 = delete
    ZENTITYPK INTEGER              -- references the affected row's Z_PK
);
