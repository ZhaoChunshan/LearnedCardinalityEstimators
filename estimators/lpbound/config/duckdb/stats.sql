-- Adapted from the parent project datasets/stats/postgres_create_stats.sql
CREATE TABLE badges (
id integer not null,
    userid integer not null,
    date bigint not null
);

CREATE TABLE comments (
id integer not null,
    postid integer not null,
    score smallint not null,
    creationdate bigint not null,
    userid integer
);

CREATE TABLE posthistory (
id integer not null,
    posthistorytypeid smallint not null,
    postid integer not null,
    creationdate bigint not null,
    userid integer
);

CREATE TABLE postlinks (
id integer not null,
    creationdate bigint not null,
    postid integer not null,
    relatedpostid integer not null,
    linktypeid smallint not null
);

CREATE TABLE posts (
id integer not null,
    posttypeid smallint not null,
    creationdate bigint not null,
    score integer,
    viewcount integer,
    owneruserid integer,
    answercount integer,
    commentcount integer,
    favoritecount integer,
    lasteditoruserid integer
);

CREATE TABLE tags (
id integer not null,
    count integer not null,
    excerptpostid integer
);

CREATE TABLE users (
id integer not null,
    reputation integer not null,
    creationdate bigint not null,
    views integer not null,
    upvotes integer not null,
    downvotes integer not null
);

CREATE TABLE votes (
id integer not null,
    postid integer not null,
    votetypeid smallint not null,
    creationdate bigint not null,
    userid integer,
    bountyamount smallint
);
