-- Adapted from the parent project datasets/ergastf1/postgres_create_ergastf1.sql
CREATE TABLE circuits (
circuitid  integer,
    circuitref varchar(255) default '' not null,
    name       varchar(255) default '' not null,
    location   varchar(255) default    null,
    country    varchar(255) default    null,
    lat        float        default    null,
    lng        float        default    null,
    alt        integer      default    null,
    url        varchar(255) default '' not null
);

CREATE TABLE constructors (
constructorid  integer,
    constructorref varchar(255) default '' not null,
    name           varchar(255) default '' not null,
    nationality    varchar(255) default    null,
    url            varchar(255) default '' not null
);

CREATE TABLE drivers (
driverid    integer,
    driverref   varchar(255) default '' not null,
    number      integer      default    null,
    code        varchar(3)   default    null,
    forename    varchar(255) default '' not null,
    surname     varchar(255) default '' not null,
    dob         VARCHAR         default    null,
    nationality integer      default 0  not null,
    url         varchar(255) default ''
);

CREATE TABLE seasons (
year integer      default 0  not null,
    url  varchar(255) default '' not null
);

CREATE TABLE races (
raceid    integer,
    year      integer      default 0            not null,
    round     integer      default 0            not null,
    circuitid integer      default 0            not null,
    name      integer      default 0            not null,
    date      VARCHAR         not null,
    time      VARCHAR                          null,
    url       varchar(255)                      null
);

CREATE TABLE constructorresults (
constructorresultsid integer,
    raceid               integer default 0 not null,
    constructorid        integer default 0 not null,
    points               float         null,
    status               varchar(255)  null
);

CREATE TABLE constructorstandings (
constructorstandingsid integer,
    raceid                 integer   default 0 not null,
    constructorid          integer   default 0 not null,
    points                 float default 0 not null,
    position               integer             null,
    positiontext           integer   default 0 not null,
    wins                   integer   default 0 not null
);

CREATE TABLE driverstandings (
driverstandingsid integer,
    raceid            integer   default 0 not null,
    driverid          integer   default 0 not null,
    points            float default 0 not null,
    position          integer             null,
    positiontext      varchar(255)    null,
    wins              integer   default 0 not null
);

CREATE TABLE laptimes (
raceid       integer          not null,
    driverid     integer          not null,
    lap          integer          not null,
    position     integer          null,
    time         varchar(255) null,
    milliseconds integer          null
);

CREATE TABLE pitstops (
raceid       integer          not null,
    driverid     integer          not null,
    stop         integer          not null,
    lap          integer          not null,
    time         VARCHAR         not null,
    duration     varchar(255) null,
    milliseconds integer          null
);

CREATE TABLE qualifying (
qualifyid     integer,
    raceid        integer default 0 not null,
    driverid      integer default 0 not null,
    constructorid integer default 0 not null,
    number        integer default 0 not null,
    position      integer           null,
    q1            varchar(255)  null,
    q2            varchar(255)  null,
    q3            varchar(255)  null
);

CREATE TABLE status (
statusid integer,
    status   varchar(255) default '' not null
);

CREATE TABLE results (
resultid        integer,
    raceid          integer          default 0  not null,
    driverid        integer          default 0  not null,
    constructorid   integer          default 0  not null,
    number          integer                     null,
    grid            integer          default 0  not null,
    position        integer                     null,
    positiontext    integer          default 0  not null,
    positionorder   integer          default 0  not null,
    points          float        default 0  not null,
    laps            integer          default 0  not null,
    time            varchar(255)            null,
    milliseconds    integer                     null,
    fastestlap      integer                     null,
    rank            integer          default 0  null,
    fastestlaptime  varchar(255)            null,
    fastestlapspeed varchar(255)            null,
    statusid        integer          default 0  not null
);

CREATE TABLE target (
targetid integer not null,
    raceid   integer not null,
    driverid integer not null,
    win      integer not null
);
