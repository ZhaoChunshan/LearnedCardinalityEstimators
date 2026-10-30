from Schemas.graph_representation import SchemaGraph, Table

def gen_stats_old_schema(hdf_path):
    """
    Generate the stats schema with a small subset of data.
    """

    schema = SchemaGraph()

    # tables

    # badges
    schema.add_table(Table('badges', attributes=['Id', 'UserId', 'Date'],
                           irrelevant_attributes=['Id'],
                           no_compression=[],
                           csv_file_location=hdf_path.format('badges'),
                           table_size=79851))

    # votes
    schema.add_table(Table('votes', attributes=['Id', 'PostId', 'VoteTypeId', 'CreationDate', 'UserId',
                                                'BountyAmount'],
                           csv_file_location=hdf_path.format('votes'),
                           irrelevant_attributes=['Id', 'BountyAmount'],
                           no_compression=['VoteTypeId'],
                           table_size=328064))

    # postHistory
    schema.add_table(Table('postHistory', attributes=['Id', 'PostHistoryTypeId', 'PostId', 'CreationDate',
                                                      'UserId'],
                           csv_file_location=hdf_path.format('postHistory'),
                           irrelevant_attributes=['Id'],
                           no_compression=['PostHistoryTypeId'],
                           table_size=303187))

    # posts
    schema.add_table(Table('posts', attributes=['Id', 'PostTypeId', 'CreationDate',
                                                'Score', 'ViewCount', 'OwnerUserId',
                                                'AnswerCount', 'CommentCount', 'FavoriteCount',
                                                'LastEditorUserId'],
                           csv_file_location=hdf_path.format('posts'),
                           irrelevant_attributes=['LastEditorUserId'],
                           no_compression=['PostTypeId'],
                           table_size=91976))

    # users
    schema.add_table(Table('users', attributes=['Id', 'Reputation', 'CreationDate', 'Views', 'UpVotes', 'DownVotes'],
                           csv_file_location=hdf_path.format('users'),
                           no_compression=[],
                           table_size=40325))

    # comments
    schema.add_table(Table('comments', attributes=['Id', 'PostId', 'Score', 'CreationDate', 'UserId'],
                           csv_file_location=hdf_path.format('comments'),
                           irrelevant_attributes=['Id'],
                           no_compression=[],
                           table_size=174305))

    # postLinks
    schema.add_table(Table('postLinks', attributes=['Id', 'CreationDate', 'PostId', 'RelatedPostId', 'LinkTypeId'],
                           csv_file_location=hdf_path.format('postLinks'),
                           irrelevant_attributes=['Id'],
                           no_compression=["LinkTypeId"],
                           table_size=11102))

    # tags
    schema.add_table(Table('tags', attributes=['Id', 'Count', 'ExcerptPostId'],
                           csv_file_location=hdf_path.format('tags'),
                           irrelevant_attributes=[],
                           no_compression=["LinkTypeId"],
                           table_size=1032))


    # relationships
    schema.add_relationship('comments', 'PostId', 'posts', 'Id')
    schema.add_relationship('comments', 'UserId', 'users', 'Id')

    schema.add_relationship('badges', 'UserId', 'users', 'Id')

    schema.add_relationship('tags', 'ExcerptPostId', 'posts', 'Id')

    schema.add_relationship('postLinks', 'PostId', 'posts', 'Id')
    schema.add_relationship('postLinks', 'RelatedPostId', 'posts', 'Id')

    schema.add_relationship('postHistory', 'PostId', 'posts', 'Id')
    schema.add_relationship('postHistory', 'UserId', 'users', 'Id')

    schema.add_relationship('votes', 'PostId', 'posts', 'Id')
    schema.add_relationship('votes', 'UserId', 'users', 'Id')

    schema.add_relationship('posts', 'OwnerUserId', 'users', 'Id')
    schema.add_relationship('posts', 'LastEditorUserId', 'users', 'Id')

    return schema


def gen_stats_light_schema(csv_path):
    """
    Generate the stats schema with a small subset of data.
    """

    schema = SchemaGraph()

    # tables

    # badges
    schema.add_table(Table('badges', attributes=['id', 'userid', 'date'],
                           irrelevant_attributes=[],
                           no_compression=[],
                           csv_file_location=csv_path.format('badges'),
                           table_size=79851))
    
    # comments
    schema.add_table(Table('comments', attributes=['id', 'postid', 'score', 'creationdate', 'userid'],
                           irrelevant_attributes=[],
                           no_compression=['score'],
                           csv_file_location=csv_path.format('comments'),
                           table_size=174305))
    
    # posthistory
    schema.add_table(Table('posthistory', attributes=['id', 'posthistorytypeid', 'postid', 'creationdate', 'userid'],
                           irrelevant_attributes=[],
                           no_compression=['posthistorytypeid'],
                           csv_file_location=csv_path.format('posthistory'),
                           table_size=303187))
    
    # postlinks
    schema.add_table(Table('postlinks', attributes=['id', 'creationdate', 'postid', 'relatedpostid', 'linktypeid'],
                           irrelevant_attributes=[],
                           no_compression=['linktypeid'],
                           csv_file_location=csv_path.format('postlinks'),
                           table_size=11102)) 
    
    # posts
    schema.add_table(Table('posts', attributes=['id', 'posttypeid', 'creationdate', 'score', 'viewcount', 'owneruserid',
                                                'answercount', 'commentcount', 'favoritecount', 'lasteditoruserid'],
                            irrelevant_attributes=[],#['lasteditoruserid'],
                            no_compression=['posttypeid', 'score', 'answercount', 'commentcount', 'favoritecount'],
                            csv_file_location=csv_path.format('posts'),
                            table_size=91976))
    
    # tags
    schema.add_table(Table('tags', attributes=['id', 'count', 'excerptpostid'],
                           irrelevant_attributes=['count'],
                           no_compression=[],
                           csv_file_location=csv_path.format('tags'),
                           table_size=1032))
    
    # users
    schema.add_table(Table('users', attributes=['id', 'reputation', 'creationdate', 'views', 'upvotes', 'downvotes'],
                           irrelevant_attributes=[],
                           no_compression=['downvotes'],
                           csv_file_location=csv_path.format('users'),
                           table_size=40325))
    
    # votes
    schema.add_table(Table('votes', attributes=['id', 'postid', 'votetypeid', 'creationdate', 'userid', 'bountyamount'],
                           irrelevant_attributes=[],
                           no_compression=['votetypeid', 'bountyamount'],
                           csv_file_location=csv_path.format('votes'),
                           table_size=328064))
    
    # relationships
    schema.add_relationship('badges', 'userid', 'users', 'id')
    schema.add_relationship('comments', 'postid', 'posts', 'id')
    schema.add_relationship('comments', 'userid', 'users', 'id')
    schema.add_relationship('posthistory', 'postid', 'posts', 'id')
    schema.add_relationship('posthistory', 'userid', 'users', 'id')
    schema.add_relationship('postlinks', 'postid', 'posts', 'id')
    schema.add_relationship('postlinks', 'relatedpostid', 'posts', 'id')
    schema.add_relationship('posts', 'owneruserid', 'users', 'id')
    schema.add_relationship('posts', 'lasteditoruserid', 'users', 'id')
    schema.add_relationship('tags', 'excerptpostid', 'posts', 'id')
    schema.add_relationship('votes', 'postid', 'posts', 'id')
    schema.add_relationship('votes', 'userid', 'users', 'id')

    return schema



