from ensemble_compilation.graph_representation import SchemaGraph, Table


def gen_stats_schema(csv_path):
    """
    Schema for STATS to do our experiment.
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
                            irrelevant_attributes=[],
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
