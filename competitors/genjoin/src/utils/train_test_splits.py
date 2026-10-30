# Licensed under GNU GPL version 3; see competitors/genjoin/LICENSE.
# Modified for the experimental study in this repository.
# Local version recorded: 2026-10-10 (project import commit 9f7a0cf).
# Licensing notice added: 2026-10-11.
# Original project: https://github.com/edualc/genjoin
# Changes: load generated training query IDs from JSONL and use the study
# workload splits.


# ===========================================================================================================
# JOIN ORDER BENCHMARK (JOB) SPLITS
# ===========================================================================================================


RANDOM_SPLIT_1__TEST_QUERIES = [
    '1c.sql', '2c.sql', '4b.sql', '4c.sql', '5c.sql', '6a.sql', '6c.sql', '6e.sql', '8b.sql', '8c.sql', '9c.sql',
    '11d.sql', '15a.sql', '17b.sql', '17e.sql', '18b.sql',
    '20a.sql', '21a.sql', '25c.sql', '28b.sql',
    '32b.sql', '33a.sql'
]
    
RANDOM_SPLIT_2__TEST_QUERIES = [
    '1a.sql', '4c.sql', '5c.sql', '6c.sql', '6d.sql', '7b.sql', '8c.sql',
    '10a.sql', '11a.sql', '11d.sql', '13c.sql', '13d.sql', '15d.sql', '16a.sql', '17b.sql', '19a.sql',
    '20a.sql', '22b.sql', '25b.sql', '29b.sql',
    '31a.sql', '32b.sql'
]
    
RANDOM_SPLIT_3__TEST_QUERIES = [
    '2a.sql', '3b.sql', '6d.sql', '9b.sql',
    '10b.sql', '11b.sql', '11c.sql', '13c.sql', '13d.sql', '16b.sql', '18c.sql', '19c.sql',
    '21c.sql', '22a.sql', '22d.sql', '26a.sql', '26b.sql', '27c.sql', '28a.sql', '28c.sql',
    '30a.sql', '33c.sql'
]


BASE_QUERY_SPLIT_1__TEST_QUERIES = [
    '2a.sql', '2b.sql', '2c.sql', '2d.sql',
    '7a.sql', '7b.sql', '7c.sql',
    '15a.sql', '15b.sql', '15c.sql', '15d.sql',
    '24a.sql', '24b.sql', 
    '25a.sql', '25b.sql', '25c.sql',
    '31a.sql', '31b.sql', '31c.sql',
]

BASE_QUERY_SPLIT_2__TEST_QUERIES = [
    '13a.sql', '13b.sql', '13c.sql', '13d.sql',
    '15a.sql', '15b.sql', '15c.sql', '15d.sql',
    '20a.sql', '20b.sql', '20c.sql',
    '26a.sql', '26b.sql', '26c.sql',
    '29a.sql', '29b.sql', '29c.sql',
    '30a.sql', '30b.sql', '30c.sql',
    '33a.sql', '33b.sql', '33c.sql'
]

BASE_QUERY_SPLIT_3__TEST_QUERIES = [
    '1a.sql', '1b.sql', '1c.sql', '1d.sql',
    '5a.sql', '5b.sql', '5c.sql',
    '12a.sql', '12b.sql', '12c.sql',
    '17a.sql', '17b.sql', '17c.sql', '17d.sql', '17e.sql', '17f.sql',
    '22a.sql', '22b.sql', '22c.sql', '22d.sql',
    '27a.sql', '27b.sql', '27c.sql',
    '28a.sql', '28b.sql', '28c.sql'
]


LEAVE_ONE_OUT_SPLIT_1__TEST_QUERIES = [
    '1c.sql', '2a.sql', '3b.sql', '4a.sql', '5a.sql', '6b.sql', '7c.sql', '8c.sql', '9c.sql',
    '10b.sql', '11b.sql', '12c.sql', '13b.sql', '14a.sql', '15b.sql', '16c.sql', '17c.sql', '18b.sql', '19a.sql',
    '20c.sql', '21c.sql', '22b.sql', '23b.sql', '24a.sql', '25a.sql', '26c.sql', '27c.sql', '28a.sql', '29b.sql',
    '30a.sql', '31b.sql', '32b.sql', '33c.sql'
]

LEAVE_ONE_OUT_SPLIT_2__TEST_QUERIES = [
    '1d.sql', '2d.sql', '3a.sql', '4b.sql', '5c.sql', '6d.sql', '7a.sql', '8c.sql', '9c.sql',
    '10a.sql', '11a.sql', '12a.sql', '13d.sql', '14b.sql', '15b.sql', '16a.sql', '17f.sql', '18a.sql', '19d.sql',
    '20a.sql', '21b.sql', '22c.sql', '23b.sql', '24b.sql', '25a.sql', '26a.sql', '27b.sql', '28c.sql', '29a.sql',
    '30b.sql', '31a.sql', '32b.sql', '33b.sql'
]

LEAVE_ONE_OUT_SPLIT_3__TEST_QUERIES = [
    '1c.sql', '2d.sql', '3b.sql', '4a.sql', '5c.sql', '6d.sql', '7b.sql', '8a.sql', '9a.sql',
    '10c.sql', '11d.sql', '12a.sql', '13a.sql', '14b.sql', '15a.sql', '16d.sql', '17b.sql', '18b.sql', '19d.sql',
    '20b.sql', '21a.sql', '22a.sql', '23b.sql', '24a.sql', '25b.sql', '26a.sql', '27a.sql', '28b.sql', '29c.sql',
    '30a.sql', '31a.sql', '32a.sql', '33c.sql'
]


RANDOM_SPLIT_A__TEST_QUERIES = [
    "1a.sql", "2a.sql", "3b.sql", "3c.sql", "4a.sql", "4b.sql", "5b.sql", "6b.sql", "6f.sql", "8a.sql", "9a.sql", "9b.sql", "9d.sql", 
    "10c.sql", "11d.sql", "13a.sql", "13b.sql", "13d.sql", "15c.sql", "17d.sql", "18b.sql", "18c.sql", "19b.sql", "19c.sql", 
    "20a.sql", "22a.sql", "22c.sql", "22d.sql", "23b.sql", "26b.sql", "28c.sql", 
    "30b.sql", "30c.sql", "31c.sql", "32a.sql", "33a.sql", "33b.sql", 
]

RANDOM_SPLIT_B__TEST_QUERIES = [
    "1d.sql", "2b.sql", "2c.sql", "2d.sql", "3a.sql", "4c.sql", "5c.sql", "6a.sql", "6c.sql", "7b.sql", "7c.sql", "8b.sql", "8c.sql", 
    "10a.sql", "10b.sql", "11b.sql", "11c.sql", "12b.sql", "12c.sql", "14b.sql", "14c.sql", "15d.sql", "16a.sql", "16b.sql", "17f.sql", "18a.sql", "19a.sql", 
    "20b.sql", "22b.sql", "23c.sql", "24a.sql", "25c.sql", "26a.sql", "27a.sql", "27c.sql", "28b.sql", "29b.sql", 
]

RANDOM_SPLIT_C__TEST_QUERIES = [
    "1b.sql", "1c.sql", "5a.sql", "6d.sql", "6e.sql", "7a.sql", "8d.sql", "9c.sql", 
    "11a.sql", "12a.sql", "13c.sql", "14a.sql", "15a.sql", "15b.sql", "16c.sql", "16d.sql", "17a.sql", "17b.sql", "17c.sql", "17e.sql", "19d.sql", 
    "20c.sql", "21a.sql", "21b.sql", "21c.sql", "23a.sql", "24b.sql", "25a.sql", "25b.sql", "26c.sql", "27b.sql", "28a.sql", "29a.sql", "29c.sql", 
    "30a.sql", "31a.sql", "31b.sql", "32b.sql", "33c.sql", 
]


BASE_QUERY_SPLIT_A__TEST_QUERIES = [
    "1a.sql", "1b.sql", "1c.sql", "1d.sql", "5a.sql", "5b.sql", "5c.sql", "9a.sql", "9b.sql", "9c.sql", "9d.sql", 
    "10a.sql", "10b.sql", "10c.sql", "13a.sql", "13b.sql", "13c.sql", "13d.sql", "16a.sql", "16b.sql", "16c.sql", "16d.sql", "18a.sql", "18b.sql", "18c.sql", 
    "20a.sql", "20b.sql", "20c.sql", "22a.sql", "22b.sql", "22c.sql", "22d.sql", "27a.sql", "27b.sql", "27c.sql", 
    "32a.sql", "32b.sql", 
]

BASE_QUERY_SPLIT_B__TEST_QUERIES = [
    "2a.sql", "2b.sql", "2c.sql", "2d.sql", "3a.sql", "3b.sql", "3c.sql", "4a.sql", "4b.sql", "4c.sql", "6a.sql", "6b.sql", "6c.sql", "6d.sql", "6e.sql", "6f.sql", 
    "12a.sql", "12b.sql", "12c.sql", "14a.sql", "14b.sql", "14c.sql", "17a.sql", "17b.sql", "17c.sql", "17d.sql", "17e.sql", "17f.sql", 
    "25a.sql", "25b.sql", "25c.sql", "28a.sql", "28b.sql", "28c.sql", 
    "30a.sql", "30b.sql", "30c.sql", 
]

BASE_QUERY_SPLIT_C__TEST_QUERIES = [
    "7a.sql", "7b.sql", "7c.sql", "8a.sql", "8b.sql", "8c.sql", "8d.sql", 
    "11a.sql", "11b.sql", "11c.sql", "11d.sql", "15a.sql", "15b.sql", "15c.sql", "15d.sql", "19a.sql", "19b.sql", "19c.sql", "19d.sql", 
    "21a.sql", "21b.sql", "21c.sql", "23a.sql", "23b.sql", "23c.sql", "24a.sql", "24b.sql", "26a.sql", "26b.sql", "26c.sql", "29a.sql", "29b.sql", "29c.sql", 
    "31a.sql", "31b.sql", "31c.sql", "33a.sql", "33b.sql", "33c.sql", 
]


LEAVE_ONE_OUT_SPLIT_A__TEST_QUERIES = [
    "1b.sql", "1d.sql", "2c.sql", "3b.sql", "4a.sql", "5a.sql", "6c.sql", "6f.sql", "7b.sql", "8b.sql", "9c.sql", "9d.sql", 
    "10b.sql", "11a.sql", "12a.sql", "13b.sql", "14a.sql", "15a.sql", "15d.sql", "16c.sql", "17e.sql", "17f.sql", "18c.sql", "19b.sql", 
    "20c.sql", "21a.sql", "22c.sql", "22d.sql", "23a.sql", "24a.sql", "25c.sql", "26c.sql", "27a.sql", "28c.sql", "29b.sql", 
    "30b.sql", "31b.sql", "32a.sql", "33b.sql", 
]

LEAVE_ONE_OUT_SPLIT_B__TEST_QUERIES = [
    "1a.sql", "2b.sql", "2d.sql", "3c.sql", "4c.sql", "5c.sql", "6d.sql", "6e.sql", "7a.sql", "8d.sql", "9a.sql", 
    "10a.sql", "11b.sql", "11c.sql", "12c.sql", "13c.sql", "14b.sql", "15c.sql", "16a.sql", "16b.sql", "17a.sql", "17c.sql", "18a.sql", "19c.sql", 
    "20b.sql", "21c.sql", "22a.sql", "23c.sql", "24a.sql", "25a.sql", "26a.sql", "27c.sql", "28b.sql", "29c.sql", 
    "30a.sql", "31a.sql", "32a.sql", "33a.sql", 
]

LEAVE_ONE_OUT_SPLIT_C__TEST_QUERIES = [
    "1c.sql", "2a.sql", "3a.sql", "4b.sql", "5b.sql", "6a.sql", "6b.sql", "7c.sql", "8a.sql", "8c.sql", "9b.sql", 
    "10c.sql", "11d.sql", "12b.sql", "13a.sql", "13d.sql", "14c.sql", "15b.sql", "16d.sql", "17b.sql", "17d.sql", "18b.sql", "19a.sql", "19d.sql", 
    "20a.sql", "21b.sql", "22b.sql", "23b.sql", "24b.sql", "25b.sql", "26b.sql", "27b.sql", "28a.sql", "29a.sql", 
    "30c.sql", "31c.sql", "32b.sql", "33c.sql", 
]

SLOW_SPLIT__TEST_QUERIES = [
    '16b.sql', '17a.sql', '17e.sql', '17f.sql', '17b.sql', '19d.sql', '17d.sql',
    '17c.sql', '10c.sql', '26c.sql', '25c.sql', '6d.sql', '6f.sql', '8c.sql',
    '18c.sql', '9d.sql', '30a.sql', '19c.sql', '20a.sql'
]



# ===========================================================================================================
# STACK SPLITS
# ===========================================================================================================


STACK__RANDOM_SPLIT_1__TEST_QUERIES = [
    'q1__q1-031.sql', 'q1__q1-035.sql', 'q4__q4-042.sql', 'q4__q4-064.sql',
    'q4__q4-089.sql', 'q5__q5-032.sql', 'q6__q6-060.sql', 'q6__q6-064.sql', 'q7__q7-099.sql',
    'q11__6c5cba419c5b7b02d431aeb5e766d775d812967a.sql', 'q11__c1ae2a992cde4ea2c4922d852df22043254b4f84.sql',
    'q12__55de941e8497cfeeb93d3f8f2d7a18489e0e6c32.sql', 'q14__63c0776f1727638316b966fe748df7cc585a335b.sql',
    'q14__74fd1af68d23f0690e3d0fc80bd9b42fa90a7e94.sql', 'q14__97e68ad5c2ced4c182366b3118a1f5f69b423fa6.sql',
    'q14__b49361f85785200ed6ec1f2eec357b7598c9e564.sql', 'q15__3e37e62655ceaebc14e79edad518e5710752f51d.sql',
    'q15__543ab3f730e494a69e3d15e59675f491544cb15d.sql', 'q15__b8ddf65b0c0c7867a9b560e571d457fec410715c.sql',
    'q15__d5546c01928a687eb1f54e9f8eb4e1aff68fc381.sql', 'q16__1e863562a79ca1f7754c759ebab6a2addda0bde8.sql',
    'q16__ea9efde510227beb8d624b8c4a6941b9d5e6e637.sql', 'q16__ed2ffeaefcf5ad8bbadc713ccc766541e12080aa.sql'
]

STACK__RANDOM_SPLIT_2__TEST_QUERIES = [
    'q1__q1-009.sql', 'q1__q1-031.sql', 'q1__q1-035.sql', 'q1__q1-098.sql', 
    'q4__q4-089.sql', 'q5__q5-059.sql', 'q5__q5-082.sql', 'q6__q6-065.sql', 'q7__q7-082.sql', 'q7__q7-085.sql', 
    'q8__q8-062.sql', 'q8__q8-096.sql', 'q11__9389f58853715321e2a60ad743f99fc365f040cb.sql', 
    'q14__719e692d411868ae7a93909757872d264f6bbf73.sql', 'q14__74fd1af68d23f0690e3d0fc80bd9b42fa90a7e94.sql', 
    'q14__b49361f85785200ed6ec1f2eec357b7598c9e564.sql', 'q15__b2ee2c788d30655058aeb992811e9a54f17f2998.sql', 
    'q15__d5546c01928a687eb1f54e9f8eb4e1aff68fc381.sql', 'q16__1e863562a79ca1f7754c759ebab6a2addda0bde8.sql', 
    'q16__374e3e4c9eefc294fa4c46220953336298df3622.sql', 'q16__d5290889129fb8e625f2b36fa106e30d6c4b243b.sql', 
    'q16__ea9efde510227beb8d624b8c4a6941b9d5e6e637.sql', 'q16__fbe34e8fdf672a34fd82cbbd6d9a81fd02ce17d1.sql'
]

STACK__RANDOM_SPLIT_3__TEST_QUERIES = [
    'q1__q1-009.sql', 'q1__q1-031.sql', 'q2__q2-001.sql', 'q2__q2-035.sql', 'q2__q2-098.sql', 'q3__q3-086.sql', 
    'q4__q4-086.sql', 'q5__q5-015.sql', 'q5__q5-052.sql', 'q5__q5-079.sql', 'q6__q6-065.sql', 'q6__q6-085.sql', 
    'q7__q7-085.sql', 'q8__q8-006.sql', 'q8__q8-025.sql', 'q11__0ea8bacde0e13a4314466435cf49c8e685b39fb1.sql', 
    'q12__5a5ff9bd9de9e748708116727803117e453e30da.sql', 'q12__76a47868a09eec9f95bacb2cf21492d353698eb7.sql', 
    'q13__13ad1b8c6bea4fda1892b9fa82cc1ceb9ceb85fc.sql', 'q14__4063b6cbbd1c0f2a902a647aafe24174a75f53cd.sql', 
    'q14__97e68ad5c2ced4c182366b3118a1f5f69b423fa6.sql', 'q16__d5290889129fb8e625f2b36fa106e30d6c4b243b.sql', 
    'q16__ea9efde510227beb8d624b8c4a6941b9d5e6e637.sql'
]

STACK__BASE_QUERY_SPLIT_1__TEST_QUERIES = [
    'q2__q2-001.sql', 'q2__q2-012.sql', 'q2__q2-032.sql', 'q2__q2-035.sql',
    'q2__q2-050.sql', 'q2__q2-081.sql', 'q2__q2-094.sql', 'q2__q2-098.sql',
    'q7__q7-034.sql', 'q7__q7-036.sql', 'q7__q7-047.sql', 'q7__q7-077.sql',
    'q7__q7-082.sql', 'q7__q7-085.sql', 'q7__q7-095.sql', 'q7__q7-099.sql',
    'q13__13ad1b8c6bea4fda1892b9fa82cc1ceb9ceb85fc.sql', 'q13__1ddcc8650e17b292bc7344902baffc90c5ae5761.sql',
    'q13__935e2051bf80eeafe91aeb6eb719b6b64b9592c2.sql', 'q13__a091adce62743b65c04532e98e8ff3d7e546ea77.sql',
    'q13__a3d03772d880754fc4e150d82908757477ae2186.sql', 'q13__add0df9dccb2790c14508e19c9e0deb79fad6ea2.sql',
    'q13__d383cd5b4aee7d3f73508e2a1fe5f6d0f7dd42a2.sql', 'q13__d4707be2adfdbc842f42acb1fc16e3a43faf7474.sql'
]

STACK__BASE_QUERY_SPLIT_2__TEST_QUERIES = [
    'q3__q3-018.sql', 'q3__q3-040.sql', 'q3__q3-043.sql', 'q3__q3-046.sql', 'q3__q3-066.sql', 'q3__q3-068.sql', 
    'q3__q3-086.sql', 'q3__q3-099.sql', 'q6__q6-002.sql', 'q6__q6-009.sql', 'q6__q6-060.sql', 'q6__q6-064.sql', 
    'q6__q6-065.sql', 'q6__q6-067.sql', 'q6__q6-069.sql', 'q6__q6-085.sql', 'q8__q8-006.sql', 'q8__q8-025.sql', 
    'q8__q8-046.sql', 'q8__q8-062.sql', 'q8__q8-065.sql', 'q8__q8-074.sql', 'q8__q8-076.sql', 'q8__q8-096.sql'
]

STACK__BASE_QUERY_SPLIT_3__TEST_QUERIES = [
    'q1__q1-009.sql', 'q1__q1-031.sql', 'q1__q1-035.sql', 'q1__q1-067.sql', 'q1__q1-075.sql', 'q1__q1-098.sql', 
    'q1__q1-099.sql', 'q1__q1-100.sql', 'q4__q4-002.sql', 'q4__q4-026.sql', 'q4__q4-041.sql', 'q4__q4-042.sql', 
    'q4__q4-064.sql', 'q4__q4-074.sql', 'q4__q4-086.sql', 'q4__q4-089.sql', 'q6__q6-002.sql', 'q6__q6-009.sql', 
    'q6__q6-060.sql', 'q6__q6-064.sql', 'q6__q6-065.sql', 'q6__q6-067.sql', 'q6__q6-069.sql', 'q6__q6-085.sql'
]

STACK__LEAVE_ONE_OUT_SPLIT_1__TEST_QUERIES = [
    'q1__q1-035.sql', 'q2__q2-032.sql', 'q3__q3-043.sql', 'q4__q4-041.sql',
    'q5__q5-041.sql', 'q6__q6-060.sql', 'q7__q7-047.sql', 'q8__q8-046.sql',
    'q11__6c5cba419c5b7b02d431aeb5e766d775d812967a.sql', 'q12__547c6bf1994c9b2ba82a7ae32f4b051beabf46fd.sql',
    'q13__935e2051bf80eeafe91aeb6eb719b6b64b9592c2.sql', 'q14__5e4835cd72aaa2d7be15b2a5ffa2e66156b3656f.sql',
    'q15__543ab3f730e494a69e3d15e59675f491544cb15d.sql', 'q16__b1a96cd48ba297dd93bce73c27b491069ad7449f.sql'
]

STACK__LEAVE_ONE_OUT_SPLIT_2__TEST_QUERIES = [
    'q1__q1-098.sql', 'q2__q2-081.sql', 'q3__q3-068.sql', 'q4__q4-074.sql', 'q5__q5-077.sql', 'q6__q6-067.sql', 
    'q7__q7-085.sql', 'q8__q8-074.sql', 'q11__aa96c8d7abbf8a5b6d29473c1b9447a84f8b4f52.sql', 
    'q12__76a47868a09eec9f95bacb2cf21492d353698eb7.sql', 'q13__add0df9dccb2790c14508e19c9e0deb79fad6ea2.sql', 
    'q14__74fd1af68d23f0690e3d0fc80bd9b42fa90a7e94.sql', 'q15__b8ddf65b0c0c7867a9b560e571d457fec410715c.sql', 
    'q16__ed2ffeaefcf5ad8bbadc713ccc766541e12080aa.sql'
]

STACK__LEAVE_ONE_OUT_SPLIT_3__TEST_QUERIES = [
    'q1__q1-067.sql', 'q2__q2-035.sql', 'q3__q3-046.sql', 'q4__q4-042.sql', 'q5__q5-052.sql', 'q6__q6-064.sql', 
    'q7__q7-077.sql', 'q8__q8-062.sql', 'q11__87c4bd0930b02a3361ac2e86c453db1fec60dc6b.sql', 
    'q12__55de941e8497cfeeb93d3f8f2d7a18489e0e6c32.sql', 'q13__a091adce62743b65c04532e98e8ff3d7e546ea77.sql', 
    'q14__63c0776f1727638316b966fe748df7cc585a335b.sql', 'q15__78995a5fc0536aa53b99be32ce84dcbf40e826f3.sql', 
    'q16__d5290889129fb8e625f2b36fa106e30d6c4b243b.sql'
]


STACK__RANDOM_SPLIT_A__TEST_QUERIES = [
    "q1__q1-009.sql", "q1__q1-075.sql", "q2__q2-012.sql", "q2__q2-032.sql", "q2__q2-035.sql", "q2__q2-050.sql", 
    "q2__q2-098.sql", "q3__q3-043.sql", "q3__q3-086.sql", "q4__q4-041.sql", "q4__q4-086.sql", "q4__q4-089.sql", 
    "q5__q5-032.sql", "q5__q5-059.sql", "q6__q6-002.sql", "q6__q6-060.sql", "q6__q6-065.sql", "q6__q6-067.sql", 
    "q6__q6-085.sql", "q7__q7-085.sql", "q7__q7-099.sql", "q8__q8-006.sql", "q8__q8-076.sql", "q11__0ea8bacde0e13a4314466435cf49c8e685b39fb1.sql", 
    "q11__33e1caf220e5bea2e592c82eede1c0427e2c2570.sql", "q11__6c5cba419c5b7b02d431aeb5e766d775d812967a.sql", 
    "q11__9389f58853715321e2a60ad743f99fc365f040cb.sql", "q11__aa96c8d7abbf8a5b6d29473c1b9447a84f8b4f52.sql", 
    "q12__0700720596313f7fa30c0dd3d4a3001c896ba760.sql", "q12__76a47868a09eec9f95bacb2cf21492d353698eb7.sql", 
    "q13__13ad1b8c6bea4fda1892b9fa82cc1ceb9ceb85fc.sql", "q13__1ddcc8650e17b292bc7344902baffc90c5ae5761.sql", 
    "q13__add0df9dccb2790c14508e19c9e0deb79fad6ea2.sql", "q14__5e4835cd72aaa2d7be15b2a5ffa2e66156b3656f.sql", 
    "q14__63c0776f1727638316b966fe748df7cc585a335b.sql", "q15__3e37e62655ceaebc14e79edad518e5710752f51d.sql",
    "q15__d5546c01928a687eb1f54e9f8eb4e1aff68fc381.sql" 
]

STACK__RANDOM_SPLIT_B__TEST_QUERIES = [
    "q1__q1-067.sql", "q1__q1-098.sql", "q1__q1-099.sql", "q1__q1-100.sql", "q2__q2-001.sql", "q2__q2-081.sql", 
    "q3__q3-018.sql", "q3__q3-040.sql", "q3__q3-046.sql", "q4__q4-002.sql", "q4__q4-026.sql", "q4__q4-042.sql", 
    "q4__q4-064.sql", "q5__q5-041.sql", "q5__q5-052.sql", "q5__q5-079.sql", "q5__q5-082.sql", "q6__q6-064.sql", 
    "q7__q7-036.sql", "q7__q7-047.sql", "q7__q7-095.sql", "q11__87c4bd0930b02a3361ac2e86c453db1fec60dc6b.sql", 
    "q11__c1ae2a992cde4ea2c4922d852df22043254b4f84.sql", "q12__06c8d6886a03d4d92837f38ff395b888de007d33.sql", 
    "q12__812a3effb91cb789490fc2e12af772b1a35f8552.sql", "q12__bde6c0cf5e67ddae4ec0dbb787291da703e406d5.sql", 
    "q13__a091adce62743b65c04532e98e8ff3d7e546ea77.sql", "q13__a3d03772d880754fc4e150d82908757477ae2186.sql", 
    "q14__4063b6cbbd1c0f2a902a647aafe24174a75f53cd.sql", "q14__5dbc1d1f1a0467ad0086e6cb337144387a37533a.sql", 
    "q14__97e68ad5c2ced4c182366b3118a1f5f69b423fa6.sql", "q15__21e4988a3f47be288de5891d69acf91928ed94eb.sql", 
    "q15__78995a5fc0536aa53b99be32ce84dcbf40e826f3.sql", "q15__b8ddf65b0c0c7867a9b560e571d457fec410715c.sql", 
    "q16__374e3e4c9eefc294fa4c46220953336298df3622.sql", "q16__d5290889129fb8e625f2b36fa106e30d6c4b243b.sql",
    "q16__f67cec3d635586efb847c832072be83b42cc45b7.sql" 
]

STACK__RANDOM_SPLIT_C__TEST_QUERIES = [
    "q1__q1-031.sql", "q1__q1-035.sql", "q2__q2-094.sql", "q3__q3-066.sql", "q3__q3-068.sql", "q3__q3-099.sql", 
    "q4__q4-074.sql", "q5__q5-015.sql", "q5__q5-077.sql", "q6__q6-009.sql", "q6__q6-069.sql", "q7__q7-034.sql", 
    "q7__q7-077.sql", "q7__q7-082.sql", "q8__q8-025.sql", "q8__q8-046.sql", "q8__q8-062.sql", "q8__q8-065.sql", 
    "q8__q8-074.sql", "q8__q8-096.sql", "q11__e4ca35591923cf4efc89e64b17fb4d330c0b34df.sql", "q12__547c6bf1994c9b2ba82a7ae32f4b051beabf46fd.sql", 
    "q12__55de941e8497cfeeb93d3f8f2d7a18489e0e6c32.sql", "q12__5a5ff9bd9de9e748708116727803117e453e30da.sql", 
    "q13__935e2051bf80eeafe91aeb6eb719b6b64b9592c2.sql", "q13__d383cd5b4aee7d3f73508e2a1fe5f6d0f7dd42a2.sql", 
    "q13__d4707be2adfdbc842f42acb1fc16e3a43faf7474.sql", "q14__719e692d411868ae7a93909757872d264f6bbf73.sql", 
    "q14__74fd1af68d23f0690e3d0fc80bd9b42fa90a7e94.sql", "q14__b49361f85785200ed6ec1f2eec357b7598c9e564.sql", 
    "q15__543ab3f730e494a69e3d15e59675f491544cb15d.sql", "q15__b2ee2c788d30655058aeb992811e9a54f17f2998.sql", 
    "q15__c9619ad44302bada330d337c174f9dab77538622.sql", "q16__1e863562a79ca1f7754c759ebab6a2addda0bde8.sql", 
    "q16__b1a96cd48ba297dd93bce73c27b491069ad7449f.sql", "q16__ea9efde510227beb8d624b8c4a6941b9d5e6e637.sql", 
    "q16__ed2ffeaefcf5ad8bbadc713ccc766541e12080aa.sql", "q16__fbe34e8fdf672a34fd82cbbd6d9a81fd02ce17d1.sql", 
]


STACK__BASE_QUERY_SPLIT_A__TEST_QUERIES = [
    "q1__q1-009.sql", "q1__q1-031.sql", "q1__q1-035.sql", "q1__q1-067.sql", "q1__q1-075.sql", "q1__q1-098.sql", 
    "q1__q1-099.sql", "q1__q1-100.sql", "q6__q6-002.sql", "q6__q6-009.sql", "q6__q6-060.sql", "q6__q6-064.sql", 
    "q6__q6-065.sql", "q6__q6-067.sql", "q6__q6-069.sql", "q6__q6-085.sql", "q12__06c8d6886a03d4d92837f38ff395b888de007d33.sql", 
    "q12__0700720596313f7fa30c0dd3d4a3001c896ba760.sql", "q12__547c6bf1994c9b2ba82a7ae32f4b051beabf46fd.sql", 
    "q12__55de941e8497cfeeb93d3f8f2d7a18489e0e6c32.sql", "q12__5a5ff9bd9de9e748708116727803117e453e30da.sql", 
    "q12__76a47868a09eec9f95bacb2cf21492d353698eb7.sql", "q12__812a3effb91cb789490fc2e12af772b1a35f8552.sql", 
    "q12__bde6c0cf5e67ddae4ec0dbb787291da703e406d5.sql", "q14__4063b6cbbd1c0f2a902a647aafe24174a75f53cd.sql", 
    "q14__5dbc1d1f1a0467ad0086e6cb337144387a37533a.sql", "q14__5e4835cd72aaa2d7be15b2a5ffa2e66156b3656f.sql", 
    "q14__63c0776f1727638316b966fe748df7cc585a335b.sql", "q14__719e692d411868ae7a93909757872d264f6bbf73.sql", 
    "q14__74fd1af68d23f0690e3d0fc80bd9b42fa90a7e94.sql", "q14__97e68ad5c2ced4c182366b3118a1f5f69b423fa6.sql", 
    "q14__b49361f85785200ed6ec1f2eec357b7598c9e564.sql", "q15__21e4988a3f47be288de5891d69acf91928ed94eb.sql", 
    "q15__3e37e62655ceaebc14e79edad518e5710752f51d.sql", "q15__543ab3f730e494a69e3d15e59675f491544cb15d.sql", 
    "q15__78995a5fc0536aa53b99be32ce84dcbf40e826f3.sql", "q15__b2ee2c788d30655058aeb992811e9a54f17f2998.sql", 
    "q15__b8ddf65b0c0c7867a9b560e571d457fec410715c.sql", "q15__c9619ad44302bada330d337c174f9dab77538622.sql",
    "q15__d5546c01928a687eb1f54e9f8eb4e1aff68fc381.sql" 
]

STACK__BASE_QUERY_SPLIT_B__TEST_QUERIES = [
    "q2__q2-001.sql", "q2__q2-012.sql", "q2__q2-032.sql", "q2__q2-035.sql", "q2__q2-050.sql", "q2__q2-081.sql", 
    "q2__q2-094.sql", "q2__q2-098.sql", "q3__q3-018.sql", "q3__q3-040.sql", "q3__q3-043.sql", "q3__q3-046.sql", 
    "q3__q3-066.sql", "q3__q3-068.sql", "q3__q3-086.sql", "q3__q3-099.sql", "q5__q5-015.sql", "q5__q5-032.sql", 
    "q5__q5-041.sql", "q5__q5-052.sql", "q5__q5-059.sql", "q5__q5-077.sql", "q5__q5-079.sql", "q5__q5-082.sql", 
    "q11__0ea8bacde0e13a4314466435cf49c8e685b39fb1.sql", "q11__33e1caf220e5bea2e592c82eede1c0427e2c2570.sql", 
    "q11__6c5cba419c5b7b02d431aeb5e766d775d812967a.sql", "q11__87c4bd0930b02a3361ac2e86c453db1fec60dc6b.sql", 
    "q11__9389f58853715321e2a60ad743f99fc365f040cb.sql", "q11__aa96c8d7abbf8a5b6d29473c1b9447a84f8b4f52.sql", 
    "q11__c1ae2a992cde4ea2c4922d852df22043254b4f84.sql", "q11__e4ca35591923cf4efc89e64b17fb4d330c0b34df.sql", 
    "q16__1e863562a79ca1f7754c759ebab6a2addda0bde8.sql", "q16__374e3e4c9eefc294fa4c46220953336298df3622.sql", 
    "q16__b1a96cd48ba297dd93bce73c27b491069ad7449f.sql", "q16__d5290889129fb8e625f2b36fa106e30d6c4b243b.sql", 
    "q16__ea9efde510227beb8d624b8c4a6941b9d5e6e637.sql", "q16__ed2ffeaefcf5ad8bbadc713ccc766541e12080aa.sql", 
    "q16__f67cec3d635586efb847c832072be83b42cc45b7.sql", "q16__fbe34e8fdf672a34fd82cbbd6d9a81fd02ce17d1.sql", 
]

STACK__BASE_QUERY_SPLIT_C__TEST_QUERIES = [
    "q4__q4-002.sql", "q4__q4-026.sql", "q4__q4-041.sql", "q4__q4-042.sql", "q4__q4-064.sql", "q4__q4-074.sql", 
    "q4__q4-086.sql", "q4__q4-089.sql", "q7__q7-034.sql", "q7__q7-036.sql", "q7__q7-047.sql", "q7__q7-077.sql", 
    "q7__q7-082.sql", "q7__q7-085.sql", "q7__q7-095.sql", "q7__q7-099.sql", "q8__q8-006.sql", "q8__q8-025.sql", 
    "q8__q8-046.sql", "q8__q8-062.sql", "q8__q8-065.sql", "q8__q8-074.sql", "q8__q8-076.sql", "q8__q8-096.sql", 
    "q13__13ad1b8c6bea4fda1892b9fa82cc1ceb9ceb85fc.sql", "q13__1ddcc8650e17b292bc7344902baffc90c5ae5761.sql", 
    "q13__935e2051bf80eeafe91aeb6eb719b6b64b9592c2.sql", "q13__a091adce62743b65c04532e98e8ff3d7e546ea77.sql", 
    "q13__a3d03772d880754fc4e150d82908757477ae2186.sql", "q13__add0df9dccb2790c14508e19c9e0deb79fad6ea2.sql", 
    "q13__d383cd5b4aee7d3f73508e2a1fe5f6d0f7dd42a2.sql", "q13__d4707be2adfdbc842f42acb1fc16e3a43faf7474.sql", 
]


STACK__LEAVE_ONE_OUT_SPLIT_A__TEST_QUERIES = [
    "q1__q1-009.sql", "q1__q1-031.sql", "q1__q1-098.sql", "q2__q2-032.sql", "q2__q2-050.sql", "q2__q2-081.sql", 
    "q3__q3-043.sql", "q3__q3-099.sql", "q4__q4-041.sql", "q4__q4-074.sql", "q4__q4-089.sql", "q5__q5-015.sql", 
    "q5__q5-041.sql", "q5__q5-077.sql", "q6__q6-009.sql", "q6__q6-064.sql", "q7__q7-036.sql", "q7__q7-077.sql", 
    "q7__q7-095.sql", "q8__q8-025.sql", "q8__q8-062.sql", "q8__q8-096.sql", "q11__0ea8bacde0e13a4314466435cf49c8e685b39fb1.sql", 
    "q11__33e1caf220e5bea2e592c82eede1c0427e2c2570.sql", "q12__06c8d6886a03d4d92837f38ff395b888de007d33.sql", 
    "q12__76a47868a09eec9f95bacb2cf21492d353698eb7.sql", "q12__bde6c0cf5e67ddae4ec0dbb787291da703e406d5.sql", 
    "q13__add0df9dccb2790c14508e19c9e0deb79fad6ea2.sql", "q13__d383cd5b4aee7d3f73508e2a1fe5f6d0f7dd42a2.sql", 
    "q13__d4707be2adfdbc842f42acb1fc16e3a43faf7474.sql", "q14__5dbc1d1f1a0467ad0086e6cb337144387a37533a.sql", 
    "q14__97e68ad5c2ced4c182366b3118a1f5f69b423fa6.sql", "q15__3e37e62655ceaebc14e79edad518e5710752f51d.sql", 
    "q15__543ab3f730e494a69e3d15e59675f491544cb15d.sql", "q15__b2ee2c788d30655058aeb992811e9a54f17f2998.sql", 
    "q16__374e3e4c9eefc294fa4c46220953336298df3622.sql", "q16__d5290889129fb8e625f2b36fa106e30d6c4b243b.sql",
    "q16__f67cec3d635586efb847c832072be83b42cc45b7.sql" 
]

STACK__LEAVE_ONE_OUT_SPLIT_B__TEST_QUERIES = [
    "q1__q1-035.sql", "q1__q1-075.sql", "q1__q1-100.sql", "q2__q2-012.sql", "q2__q2-094.sql", "q3__q3-018.sql", 
    "q3__q3-046.sql", "q3__q3-086.sql", "q4__q4-002.sql", "q4__q4-064.sql", "q4__q4-086.sql", "q5__q5-052.sql", 
    "q5__q5-059.sql", "q6__q6-002.sql", "q6__q6-065.sql", "q6__q6-085.sql", "q7__q7-047.sql", "q7__q7-082.sql", 
    "q7__q7-085.sql", "q8__q8-006.sql", "q8__q8-076.sql", "q11__6c5cba419c5b7b02d431aeb5e766d775d812967a.sql", 
    "q11__87c4bd0930b02a3361ac2e86c453db1fec60dc6b.sql", "q11__c1ae2a992cde4ea2c4922d852df22043254b4f84.sql", 
    "q12__0700720596313f7fa30c0dd3d4a3001c896ba760.sql", "q12__547c6bf1994c9b2ba82a7ae32f4b051beabf46fd.sql", 
    "q12__5a5ff9bd9de9e748708116727803117e453e30da.sql", "q13__1ddcc8650e17b292bc7344902baffc90c5ae5761.sql", 
    "q13__a091adce62743b65c04532e98e8ff3d7e546ea77.sql", "q14__4063b6cbbd1c0f2a902a647aafe24174a75f53cd.sql", 
    "q14__5e4835cd72aaa2d7be15b2a5ffa2e66156b3656f.sql", "q14__719e692d411868ae7a93909757872d264f6bbf73.sql", 
    "q15__21e4988a3f47be288de5891d69acf91928ed94eb.sql", "q15__c9619ad44302bada330d337c174f9dab77538622.sql", 
    "q15__d5546c01928a687eb1f54e9f8eb4e1aff68fc381.sql", "q16__ea9efde510227beb8d624b8c4a6941b9d5e6e637.sql",
    "q16__fbe34e8fdf672a34fd82cbbd6d9a81fd02ce17d1.sql" 
]

STACK__LEAVE_ONE_OUT_SPLIT_C__TEST_QUERIES = [
    "q1__q1-067.sql", "q1__q1-099.sql", "q2__q2-001.sql", "q2__q2-035.sql", "q2__q2-098.sql", "q3__q3-040.sql", 
    "q3__q3-066.sql", "q3__q3-068.sql", "q4__q4-026.sql", "q4__q4-042.sql", "q5__q5-032.sql", "q5__q5-079.sql", 
    "q5__q5-082.sql", "q6__q6-060.sql", "q6__q6-067.sql", "q6__q6-069.sql", "q7__q7-034.sql", "q7__q7-099.sql", 
    "q8__q8-046.sql", "q8__q8-065.sql", "q8__q8-074.sql", "q11__9389f58853715321e2a60ad743f99fc365f040cb.sql", 
    "q11__aa96c8d7abbf8a5b6d29473c1b9447a84f8b4f52.sql", "q11__e4ca35591923cf4efc89e64b17fb4d330c0b34df.sql", 
    "q12__55de941e8497cfeeb93d3f8f2d7a18489e0e6c32.sql", "q12__812a3effb91cb789490fc2e12af772b1a35f8552.sql", 
    "q13__13ad1b8c6bea4fda1892b9fa82cc1ceb9ceb85fc.sql", "q13__935e2051bf80eeafe91aeb6eb719b6b64b9592c2.sql", 
    "q13__a3d03772d880754fc4e150d82908757477ae2186.sql", "q14__63c0776f1727638316b966fe748df7cc585a335b.sql", 
    "q14__74fd1af68d23f0690e3d0fc80bd9b42fa90a7e94.sql", "q14__b49361f85785200ed6ec1f2eec357b7598c9e564.sql", 
    "q15__78995a5fc0536aa53b99be32ce84dcbf40e826f3.sql", "q15__b8ddf65b0c0c7867a9b560e571d457fec410715c.sql", 
    "q16__1e863562a79ca1f7754c759ebab6a2addda0bde8.sql", "q16__b1a96cd48ba297dd93bce73c27b491069ad7449f.sql",
    "q16__ed2ffeaefcf5ad8bbadc713ccc766541e12080aa.sql" 
]

STACK_SLOW_SPLIT__TEST_QUERIES = ['q3__q3-068.sql', 'q3__q3-043.sql',
       'q16__ed2ffeaefcf5ad8bbadc713ccc766541e12080aa.sql', 'q1__q1-099.sql',
       'q11__c1ae2a992cde4ea2c4922d852df22043254b4f84.sql', 'q1__q1-100.sql',
       'q3__q3-018.sql', 'q3__q3-086.sql', 'q1__q1-098.sql', 'q3__q3-066.sql',
       'q3__q3-040.sql', 'q14__5dbc1d1f1a0467ad0086e6cb337144387a37533a.sql',
       'q13__935e2051bf80eeafe91aeb6eb719b6b64b9592c2.sql',
       'q14__74fd1af68d23f0690e3d0fc80bd9b42fa90a7e94.sql',
       'q16__1e863562a79ca1f7754c759ebab6a2addda0bde8.sql',
       'q14__97e68ad5c2ced4c182366b3118a1f5f69b423fa6.sql',
       'q16__fbe34e8fdf672a34fd82cbbd6d9a81fd02ce17d1.sql', 'q2__q2-098.sql',
       'q16__374e3e4c9eefc294fa4c46220953336298df3622.sql']


# ===========================================================================================================
# JOIN ORDER BENCHMARK LIGHT (JOB-LIGHT) SPLITS
# ===========================================================================================================

LIGHT__RANDOM_SPLIT_A__TEST_QUERIES = [
    
    "10a.sql", "11a.sql", "11c.sql", "13c.sql", "14a.sql", "16a.sql", "16b.sql", "17b.sql", "17c.sql", 
    "1c.sql", 
    "22b.sql", "25a.sql", "25b.sql", 
    "6b.sql", "6c.sql", "6e.sql", "8a.sql", "8c.sql", "9a.sql", "9c.sql", 
  ]


LIGHT__RANDOM_SPLIT_B__TEST_QUERIES = [
    
    "11b.sql", "13a.sql", "14c.sql", "17a.sql", "18b.sql", 
    "1b.sql", "1d.sql", 
    "20a.sql", "23a.sql", "23b.sql", "26b.sql", 
    "4a.sql", "4b.sql", "5a.sql", "5b.sql", "5c.sql", "7b.sql", "7c.sql", "8b.sql", "9b.sql", 
  ]

LIGHT__RANDOM_SPLIT_C__TEST_QUERIES = [
    
    "10b.sql", "10c.sql", "13b.sql", "14b.sql", "16c.sql", "18a.sql", "19a.sql", "19b.sql", "19c.sql", 
    "1a.sql", 
    "20b.sql", "22a.sql", "26a.sql", 
    "3a.sql", "3b.sql", "3c.sql", "4c.sql", "6a.sql", "6d.sql", "7a.sql", "9d.sql", "9e.sql", 
  ]


LIGHT__BASE_QUERY_SPLIT_A__TEST_QUERIES = [
    
    "10a.sql", "10b.sql", "10c.sql", "11a.sql", "11b.sql", "11c.sql", "17a.sql", "17b.sql", "17c.sql", 
    "1a.sql", "1b.sql", "1c.sql", "1d.sql", "4a.sql", "4b.sql", "4c.sql", "6a.sql", "6b.sql", "6c.sql", "6d.sql", "6e.sql", 
  ]

LIGHT__BASE_QUERY_SPLIT_B__TEST_QUERIES = [
    
    "13a.sql", "13b.sql", "13c.sql", "14a.sql", "14b.sql", "14c.sql", 
    "20a.sql", "20b.sql", "23a.sql", "23b.sql", "26a.sql", "26b.sql", 
    "5a.sql", "5b.sql", "5c.sql", "7a.sql", "7b.sql", "7c.sql", 
  ]

LIGHT__BASE_QUERY_SPLIT_C__TEST_QUERIES = [
    
    "16a.sql", "16b.sql", "16c.sql", "18a.sql", "18b.sql", "19a.sql", "19b.sql", "19c.sql", 
    "22a.sql", "22b.sql", "25a.sql", "25b.sql", 
    "3a.sql", "3b.sql", "3c.sql", "8a.sql", "8b.sql", "8c.sql", "9a.sql", "9b.sql", "9c.sql", "9d.sql", "9e.sql", 
  ]


LIGHT__LEAVE_ONE_OUT_SPLIT_A__TEST_QUERIES = [
    
    "10c.sql", "11a.sql", "13c.sql", "14c.sql", "16c.sql", "17a.sql", "18a.sql", "19c.sql", 
    "1b.sql", "1d.sql", 
    "20a.sql", "22a.sql", "23a.sql", "25a.sql", "26b.sql", 
    "3a.sql", "4b.sql", "5a.sql", "6a.sql", "7a.sql", "8a.sql", "9a.sql", "9b.sql", 
  ]

LIGHT__LEAVE_ONE_OUT_SPLIT_B__TEST_QUERIES = [
    
    "10a.sql", "11b.sql", "13a.sql", "14b.sql", "16b.sql", "17c.sql", "18a.sql", "19a.sql", 
    "1a.sql", 
    "20b.sql", "22a.sql", "23a.sql", "25a.sql", "26a.sql", 
    "3c.sql", "4a.sql", "5c.sql", "6b.sql", "6d.sql", "7b.sql", "8b.sql", "9d.sql", "9e.sql", 
  ]

LIGHT__LEAVE_ONE_OUT_SPLIT_C__TEST_QUERIES = [
    
    "10b.sql", "11c.sql", "13b.sql", "14a.sql", "16a.sql", "17b.sql", "18b.sql", "19b.sql", 
    "1c.sql", 
    "20a.sql", "22b.sql", "23b.sql", "25b.sql", "26b.sql", 
    "3b.sql", "4c.sql", "5b.sql", "6c.sql", "6e.sql", "7c.sql", "8c.sql", "9c.sql", 
  ]


def _load_train_qids_from_jsonl(workload_name):
    """Load actually generated training query IDs from JSON LINE checkpoint."""
    import json
    from src.utils.path_helper import WORKLOADS
    schema_folder = WORKLOADS[workload_name]
    jsonl_path = f'data/demonstrations/{schema_folder}/train_data.jsonl'
    qids = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line.strip())
                qids.append(str(record['qid']))
    return qids


def get_train_test_split(experiment_name, workload_name='join_order_benchmark'):
    # New workloads: train/test are naturally separated files; no validation split.
    if workload_name in ('job_light', 'stats_ceb', 'ergastf1'):
        return _load_train_qids_from_jsonl(workload_name), []

    all_available_queries = { 'join_order_benchmark':
        ['1a', '1b', '1c', '1d', '2a', '2b', '2c', '2d', '3a', '3b', '3c', '4a', '4b', '4c',
        '5a', '5b', '5c', '6a', '6b', '6c', '6d', '6e', '6f', '7a', '7b', '7c', '8a', '8b', '8c', '8d', '9a', '9b', '9c', '9d',
        '10a', '10b', '10c', '11a', '11b', '11c', '11d', '12a', '12b', '12c', '13a', '13b', '13c', '13d', '14a', '14b', '14c',
        '15a', '15b', '15c', '15d', '16a', '16b', '16c', '16d', '17a', '17b', '17c', '17d', '17e', '17f', '18a', '18b', '18c', '19a', '19b', '19c', '19d',
        '20a', '20b', '20c', '21a', '21b', '21c', '22a', '22b', '22c', '22d', '23a', '23b', '23c', '24a', '24b',
        '25a', '25b', '25c', '26a', '26b', '26c', '27a', '27b', '27c', '28a', '28b', '28c', '29a', '29b', '29c',
        '30a', '30b', '30c', '31a', '31b', '31c', '32a', '32b', '33a', '33b', '33c' 
    ],
                             'stack': 
        ['q2__q2-012', 'q14__5e4835cd72aaa2d7be15b2a5ffa2e66156b3656f', 'q14__5dbc1d1f1a0467ad0086e6cb337144387a37533a', 'q11__e4ca35591923cf4efc89e64b17fb4d330c0b34df', 
         'q11__0ea8bacde0e13a4314466435cf49c8e685b39fb1', 'q1__q1-075', 'q3__q3-040', 'q5__q5-041', 'q11__33e1caf220e5bea2e592c82eede1c0427e2c2570', 
         'q13__935e2051bf80eeafe91aeb6eb719b6b64b9592c2', 'q6__q6-002', 'q8__q8-076', 'q12__0700720596313f7fa30c0dd3d4a3001c896ba760', 'q7__q7-099', 'q4__q4-086',
         'q3__q3-068', 'q6__q6-065', 'q7__q7-082', 'q15__b2ee2c788d30655058aeb992811e9a54f17f2998', 'q1__q1-035', 'q6__q6-060', 'q7__q7-047', 'q4__q4-041', 'q4__q4-042',
         'q2__q2-035', 'q14__63c0776f1727638316b966fe748df7cc585a335b', 'q14__b49361f85785200ed6ec1f2eec357b7598c9e564', 'q3__q3-086', 'q8__q8-006', 'q8__q8-074',
         'q3__q3-046', 'q11__6c5cba419c5b7b02d431aeb5e766d775d812967a', 'q4__q4-026', 'q5__q5-079', 'q16__1e863562a79ca1f7754c759ebab6a2addda0bde8',
         'q16__ea9efde510227beb8d624b8c4a6941b9d5e6e637', 'q8__q8-065', 'q13__d383cd5b4aee7d3f73508e2a1fe5f6d0f7dd42a2', 'q2__q2-081', 'q7__q7-034', 'q7__q7-085',
         'q12__55de941e8497cfeeb93d3f8f2d7a18489e0e6c32', 'q7__q7-077', 'q2__q2-050', 'q14__74fd1af68d23f0690e3d0fc80bd9b42fa90a7e94',
         'q14__4063b6cbbd1c0f2a902a647aafe24174a75f53cd', 'q1__q1-099', 'q1__q1-067', 'q6__q6-069', 'q2__q2-094', 'q15__c9619ad44302bada330d337c174f9dab77538622',
         'q12__547c6bf1994c9b2ba82a7ae32f4b051beabf46fd', 'q3__q3-043', 'q12__06c8d6886a03d4d92837f38ff395b888de007d33', 'q12__76a47868a09eec9f95bacb2cf21492d353698eb7',
         'q14__719e692d411868ae7a93909757872d264f6bbf73', 'q15__3e37e62655ceaebc14e79edad518e5710752f51d', 'q16__374e3e4c9eefc294fa4c46220953336298df3622',
         'q12__812a3effb91cb789490fc2e12af772b1a35f8552', 'q11__9389f58853715321e2a60ad743f99fc365f040cb', 'q8__q8-046', 'q4__q4-074',
         'q15__21e4988a3f47be288de5891d69acf91928ed94eb', 'q11__87c4bd0930b02a3361ac2e86c453db1fec60dc6b', 'q6__q6-067', 'q6__q6-064',
         'q16__b1a96cd48ba297dd93bce73c27b491069ad7449f', 'q16__fbe34e8fdf672a34fd82cbbd6d9a81fd02ce17d1', 'q1__q1-009', 'q2__q2-098',
         'q13__d4707be2adfdbc842f42acb1fc16e3a43faf7474', 'q4__q4-089', 'q16__d5290889129fb8e625f2b36fa106e30d6c4b243b', 'q8__q8-096',
         'q16__f67cec3d635586efb847c832072be83b42cc45b7', 'q7__q7-036', 'q5__q5-077', 'q13__add0df9dccb2790c14508e19c9e0deb79fad6ea2',
         'q13__13ad1b8c6bea4fda1892b9fa82cc1ceb9ceb85fc', 'q5__q5-052', 'q8__q8-062', 'q4__q4-064', 'q12__bde6c0cf5e67ddae4ec0dbb787291da703e406d5',
         'q15__b8ddf65b0c0c7867a9b560e571d457fec410715c', 'q2__q2-001', 'q5__q5-015', 'q5__q5-032', 'q1__q1-100', 'q3__q3-018', 'q1__q1-098', 'q8__q8-025', 'q3__q3-099',
         'q15__78995a5fc0536aa53b99be32ce84dcbf40e826f3', 'q2__q2-032', 'q6__q6-085', 'q15__543ab3f730e494a69e3d15e59675f491544cb15d',
         'q15__d5546c01928a687eb1f54e9f8eb4e1aff68fc381', 'q7__q7-095', 'q4__q4-002', 'q5__q5-059', 'q13__a3d03772d880754fc4e150d82908757477ae2186', 'q1__q1-031',
         'q16__ed2ffeaefcf5ad8bbadc713ccc766541e12080aa', 'q14__97e68ad5c2ced4c182366b3118a1f5f69b423fa6', 'q12__5a5ff9bd9de9e748708116727803117e453e30da',
         'q11__aa96c8d7abbf8a5b6d29473c1b9447a84f8b4f52', 'q5__q5-082', 'q11__c1ae2a992cde4ea2c4922d852df22043254b4f84', 'q6__q6-009',
         'q13__a091adce62743b65c04532e98e8ff3d7e546ea77', 'q3__q3-066', 'q13__1ddcc8650e17b292bc7344902baffc90c5ae5761'
    ],
                             'join_order_benchmark_light':
        ['10a', '10b', '10c', '11a', '11b', '11c', '13a', '13b', '13c',
       '14a', '14b', '14c', '16a', '16b', '16c', '17a', '17b', '17c',
       '18a', '18b', '19a', '19b', '19c', '1a', '1b', '1c', '1d', '20a',
       '20b', '22a', '22b', '23a', '23b', '25a', '25b', '26a', '26b',
       '3a', '3b', '3c', '4a', '4b', '4c', '5a', '5b', '5c', '6a', '6b',
       '6c', '6d', '6e', '7a', '7b', '7c', '8a', '8b', '8c', '9a', '9b',
       '9c', '9d', '9e']
                            }
    
    # blacklisted_queries = {'join_order_benchmark': [
    #     '12a', '12b', '12c', '13a', '13b', '13c', '13d', '14a', '14b', '14c', '18a', '18b', '18c',
    #     '20a', '20b', '20c', '22a', '22b', '22c', '22d', '25a', '25b', '25c', '26a', '26b', '26c', 
    #     '27a', '27b', '27c', '28a', '28b', '28c', '29a', '29b', '29c',
    #     '30a', '30b', '30c', '31a', '31b', '31c', '33a', '33b', '33c'],
    #                       'stack': [
    #     'q2__q2-001', 'q2__q2-012', 'q2__q2-032', 'q2__q2-035', 'q2__q2-050',
    #     'q2__q2-081', 'q2__q2-094', 'q2__q2-098', 'q3__q3-018', 'q3__q3-040',
    #     'q3__q3-043', 'q3__q3-046', 'q3__q3-066', 'q3__q3-068', 'q3__q3-086',
    #     'q3__q3-099', 'q7__q7-034', 'q7__q7-036', 'q7__q7-047', 'q7__q7-077',
    #     'q7__q7-082', 'q7__q7-085', 'q7__q7-095', 'q7__q7-099', 'q8__q8-006',
    #     'q8__q8-025', 'q8__q8-046', 'q8__q8-062', 'q8__q8-065', 'q8__q8-074',
    #     'q8__q8-076', 'q8__q8-096']
    #                       }
    
    blacklisted_queries = {'join_order_benchmark': [],
                          'stack': [],
                          'join_order_benchmark_light': []
                          }
                            
    if workload_name == 'join_order_benchmark':
        if experiment_name == 'LeaveOneOutSplit1':
            test_queries = LEAVE_ONE_OUT_SPLIT_1__TEST_QUERIES
        elif experiment_name == 'LeaveOneOutSplit2':
            test_queries = LEAVE_ONE_OUT_SPLIT_2__TEST_QUERIES
        elif experiment_name == 'LeaveOneOutSplit3':
            test_queries = LEAVE_ONE_OUT_SPLIT_3__TEST_QUERIES

        elif experiment_name == 'LeaveOneOutSplitA':
            test_queries = LEAVE_ONE_OUT_SPLIT_A__TEST_QUERIES
        elif experiment_name == 'LeaveOneOutSplitB':
            test_queries = LEAVE_ONE_OUT_SPLIT_B__TEST_QUERIES
        elif experiment_name == 'LeaveOneOutSplitC':
            test_queries = LEAVE_ONE_OUT_SPLIT_C__TEST_QUERIES


        elif experiment_name == 'BaseQuerySplit1':
            test_queries = BASE_QUERY_SPLIT_1__TEST_QUERIES
        elif experiment_name == 'BaseQuerySplit2':
            test_queries = BASE_QUERY_SPLIT_2__TEST_QUERIES
        elif experiment_name == 'BaseQuerySplit3':
            test_queries = BASE_QUERY_SPLIT_3__TEST_QUERIES

        elif experiment_name == 'BaseQuerySplitA':
            test_queries = BASE_QUERY_SPLIT_A__TEST_QUERIES
        elif experiment_name == 'BaseQuerySplitB':
            test_queries = BASE_QUERY_SPLIT_B__TEST_QUERIES
        elif experiment_name == 'BaseQuerySplitC':
            test_queries = BASE_QUERY_SPLIT_C__TEST_QUERIES


        elif experiment_name == 'RandomSplit1':
            test_queries = RANDOM_SPLIT_1__TEST_QUERIES
        elif experiment_name == 'RandomSplit2':
            test_queries = RANDOM_SPLIT_2__TEST_QUERIES
        elif experiment_name == 'RandomSplit3':
            test_queries = RANDOM_SPLIT_3__TEST_QUERIES

        elif experiment_name == 'RandomSplitA':
            test_queries = RANDOM_SPLIT_A__TEST_QUERIES
        elif experiment_name == 'RandomSplitB':
            test_queries = RANDOM_SPLIT_B__TEST_QUERIES
        elif experiment_name == 'RandomSplitC':
            test_queries = RANDOM_SPLIT_C__TEST_QUERIES
            
        elif experiment_name == 'SlowSplit':
            test_queries = SLOW_SPLIT__TEST_QUERIES

        elif experiment_name == 'DebugHighOrderJoin':
            # Added to see what happens for queries with large number of tables joined
            return ['29a'], ['29c']
    elif workload_name == 'stack':
        if experiment_name == 'LeaveOneOutSplit1':
            test_queries = STACK__LEAVE_ONE_OUT_SPLIT_1__TEST_QUERIES
        elif experiment_name == 'LeaveOneOutSplit2':
            test_queries = STACK__LEAVE_ONE_OUT_SPLIT_2__TEST_QUERIES
        elif experiment_name == 'LeaveOneOutSplit3':
            test_queries = STACK__LEAVE_ONE_OUT_SPLIT_3__TEST_QUERIES

        elif experiment_name == 'LeaveOneOutSplitA':
            test_queries = STACK__LEAVE_ONE_OUT_SPLIT_A__TEST_QUERIES
        elif experiment_name == 'LeaveOneOutSplitB':
            test_queries = STACK__LEAVE_ONE_OUT_SPLIT_B__TEST_QUERIES
        elif experiment_name == 'LeaveOneOutSplitC':
            test_queries = STACK__LEAVE_ONE_OUT_SPLIT_C__TEST_QUERIES


        elif experiment_name == 'BaseQuerySplit1':
            test_queries = STACK__BASE_QUERY_SPLIT_1__TEST_QUERIES
        elif experiment_name == 'BaseQuerySplit2':
            test_queries = STACK__BASE_QUERY_SPLIT_2__TEST_QUERIES
        elif experiment_name == 'BaseQuerySplit3':
            test_queries = STACK__BASE_QUERY_SPLIT_3__TEST_QUERIES

        elif experiment_name == 'BaseQuerySplitA':
            test_queries = STACK__BASE_QUERY_SPLIT_A__TEST_QUERIES
        elif experiment_name == 'BaseQuerySplitB':
            test_queries = STACK__BASE_QUERY_SPLIT_B__TEST_QUERIES
        elif experiment_name == 'BaseQuerySplitC':
            test_queries = STACK__BASE_QUERY_SPLIT_C__TEST_QUERIES


        elif experiment_name == 'RandomSplit1':
            test_queries = STACK__RANDOM_SPLIT_1__TEST_QUERIES
        elif experiment_name == 'RandomSplit2':
            test_queries = STACK__RANDOM_SPLIT_2__TEST_QUERIES
        elif experiment_name == 'RandomSplit3':
            test_queries = STACK__RANDOM_SPLIT_3__TEST_QUERIES

        elif experiment_name == 'RandomSplitA':
            test_queries = STACK__RANDOM_SPLIT_A__TEST_QUERIES
        elif experiment_name == 'RandomSplitB':
            test_queries = STACK__RANDOM_SPLIT_B__TEST_QUERIES
        elif experiment_name == 'RandomSplitC':
            test_queries = STACK__RANDOM_SPLIT_C__TEST_QUERIES
            
        elif experiment_name == 'SlowSplit':
            test_queries = STACK_SLOW_SPLIT__TEST_QUERIES
            
    elif workload_name == 'join_order_benchmark_light':
        if experiment_name == 'LeaveOneOutSplitA':
            test_queries = LIGHT__LEAVE_ONE_OUT_SPLIT_A__TEST_QUERIES
        elif experiment_name == 'LeaveOneOutSplitB':
            test_queries = LIGHT__LEAVE_ONE_OUT_SPLIT_B__TEST_QUERIES
        elif experiment_name == 'LeaveOneOutSplitC':
            test_queries = LIGHT__LEAVE_ONE_OUT_SPLIT_C__TEST_QUERIES

        elif experiment_name == 'BaseQuerySplitA':
            test_queries = LIGHT__BASE_QUERY_SPLIT_A__TEST_QUERIES
        elif experiment_name == 'BaseQuerySplitB':
            test_queries = LIGHT__BASE_QUERY_SPLIT_B__TEST_QUERIES
        elif experiment_name == 'BaseQuerySplitC':
            test_queries = LIGHT__BASE_QUERY_SPLIT_C__TEST_QUERIES

        elif experiment_name == 'RandomSplitA':
            test_queries = LIGHT__RANDOM_SPLIT_A__TEST_QUERIES
        elif experiment_name == 'RandomSplitB':
            test_queries = LIGHT__RANDOM_SPLIT_B__TEST_QUERIES
        elif experiment_name == 'RandomSplitC':
            test_queries = LIGHT__RANDOM_SPLIT_C__TEST_QUERIES

    else:
        raise NotImplementedError(f"Split {experiment_name} could not be found.")
    
    test_queries = [f.split('.sql')[0] for f in test_queries if f.split('.sql')[0] not in blacklisted_queries[workload_name]]

    train_queries = []
    for q in all_available_queries[workload_name]:
        if (q not in test_queries):  # and (q not in blacklisted_queries[workload_name]):
            train_queries.append(q)

    return train_queries, test_queries
