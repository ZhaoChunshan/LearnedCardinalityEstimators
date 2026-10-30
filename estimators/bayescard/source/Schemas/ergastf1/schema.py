from Schemas.graph_representation import SchemaGraph, Table


def gen_ergastf1_schema(csv_path):
    """
    Schema for ErgastF1 to do our experiment.
    """

    schema = SchemaGraph()

    # tables

    # circuits: not used
    schema.add_table(Table('circuits', attributes=['circuitid', 'circuitref', 'name', 'location', 
                                                   'country', 'lat', 'lng', 'alt', 'url'],
                                                   primary_key=['circuitid'],
                                                   irrelevant_attributes=['circuitref', 'name', 'location', 'country',
                                                                          'lat', 'lng', 'alt', 'url'],
                                                   no_compression=['circuitid'],
                                                   csv_file_location=csv_path.format('circuits'),
                                                   table_size=73))

    # constructorresults
    schema.add_table(Table('constructorresults', attributes=['constructorresultsid', 'raceid', 'constructorid', 'points', 'status'],
                           primary_key=['constructorresultsid'],
                           irrelevant_attributes=['status'],
                           no_compression=['points'],
                           csv_file_location=csv_path.format('constructorresults'),
                           table_size=11082))
    
    # constructors
    schema.add_table(Table('constructors', attributes=['constructorid', 'constructorref', 'name', 'nationality', 'url'],
                           primary_key=['constructorid'],
                           irrelevant_attributes=['constructorref', 'name', 'nationality', 'url'],
                           no_compression=['constructorid'],
                           csv_file_location=csv_path.format('constructors'),
                           table_size=208))
    
    # constructorstandings
    schema.add_table(Table('constructorstandings', attributes=['constructorstandingsid', 'raceid', 'constructorid',
                                                               'points', 'position', 'positiontext', 'wins'],
                           primary_key=['constructorstandingsid'],
                           irrelevant_attributes=[],
                           no_compression=['constructorid', 'position', 'positiontext', 'wins'],
                           csv_file_location=csv_path.format('constructorstandings'),
                           table_size=11836))
    
    # drivers
    schema.add_table(Table('drivers', attributes=['driverid', 'driverref', 'number', 'code', 'forename', 'surname',
                                                 'dob', 'nationality', 'url'],
                           primary_key=['driverid'],
                           irrelevant_attributes=[ 'driverref', 'number', 'code', 'forename', 'surname', 'dob', 'url'],
                           no_compression=['nationality'],
                           csv_file_location=csv_path.format('drivers'),
                           table_size=840))
    
    # driverstandings
    schema.add_table(Table('driverstandings', attributes=['driverstandingsid', 'raceid', 'driverid', 'points', 'position',
                                                            'positiontext', 'wins'],
                           primary_key=['driverstandingsid'],
                           irrelevant_attributes=['positiontext'],
                           no_compression=['position', 'wins'],
                           csv_file_location=csv_path.format('driverstandings'),
                           table_size=31578))
    

    # laptimes
    schema.add_table(Table('laptimes', attributes=['raceid', 'driverid', 'lap', 'position', 'time', 'milliseconds'],
                           primary_key=['raceid', 'driverid', 'lap'],
                           irrelevant_attributes=['time'],
                           no_compression=['driverid', 'lap', 'position'],
                           csv_file_location=csv_path.format('laptimes'),
                           table_size=420369))
    
    # pitstops
    schema.add_table(Table('pitstops', attributes=['raceid', 'driverid', 'stop', 'lap', 'time', 'duration', 'milliseconds'],
                           primary_key=['raceid', 'driverid', 'stop'],
                           irrelevant_attributes=['time', 'duration'],
                           no_compression=['raceid', 'driverid', 'stop', 'lap'],
                           csv_file_location=csv_path.format('pitstops'),
                           table_size=6070))
    
    # qualifying
    schema.add_table(Table('qualifying', attributes=['qualifyid', 'raceid', 'driverid', 'constructorid', 'number', 'position', 'q1', 'q2', 'q3'],
                           primary_key=['qualifyid'],
                           irrelevant_attributes=['q1', 'q2', 'q3'],
                           no_compression=['driverid', 'constructorid', 'number', 'position'],
                           csv_file_location=csv_path.format('qualifying'),
                           table_size=7397))
    
    # races
    schema.add_table(Table('races', attributes=['raceid', 'year', 'round', 'circuitid', 'name', 'date', 'time', 'url'],
                           primary_key=['raceid'],
                           irrelevant_attributes=['date', 'time', 'url'],
                           no_compression=['year', 'round', 'circuitid', 'name'],
                           csv_file_location=csv_path.format('races'),
                           table_size=976))
    
    # results
    schema.add_table(Table('results', attributes=['resultid', 'raceid', 'driverid', 'constructorid', 'number', 'grid', 'position', 'positiontext',
                                                  'positionorder', 'points', 'laps', 'time', 'milliseconds', 'fastestlap', 'rank', 'fastestlaptime',
                                                  'fastestlapspeed', 'statusid'],
                           primary_key=['resultid'],
                           irrelevant_attributes=['time', 'fastestlaptime', 'fastestlapspeed'],
                           no_compression=['constructorid', 'number', 'grid', 'position', 'positiontext',
                                           'positionorder', 'points', 'laps', 'fastestlap', 'rank', 'statusid'],
                           csv_file_location=csv_path.format('results'),
                           table_size=23657))
    
    # seasons
    schema.add_table(Table('seasons', attributes=['year', 'url'],
                           primary_key=['year'],
                           irrelevant_attributes=['url'],
                           no_compression=['year'],
                           csv_file_location=csv_path.format('seasons'),
                           table_size=68))
    
    # status
    schema.add_table(Table('status', attributes=['statusid', 'status'],
                           primary_key=['statusid'],
                           irrelevant_attributes=['status'],
                           no_compression=['statusid'],
                           csv_file_location=csv_path.format('status'),
                           table_size=134))
    
    # target
    schema.add_table(Table('target', attributes=['targetid', 'raceid', 'driverid', 'win'],
                           primary_key=['targetid'],
                           irrelevant_attributes=[],
                           no_compression=['win'],
                           csv_file_location=csv_path.format('target'),
                           table_size=31578))
    
    # relationships
    schema.add_relationship('races', 'circuitid', 'circuits', 'circuitid')
    schema.add_relationship('races', 'year', 'seasons', 'year')
    schema.add_relationship('constructorresults', 'raceid', 'races', 'raceid')
    schema.add_relationship('constructorresults', 'constructorid', 'constructors', 'constructorid')
    schema.add_relationship('constructorstandings', 'raceid', 'races', 'raceid')
    schema.add_relationship('constructorstandings', 'constructorid', 'constructors', 'constructorid')
    schema.add_relationship('driverstandings', 'raceid', 'races', 'raceid')
    schema.add_relationship('driverstandings', 'driverid', 'drivers', 'driverid')
    schema.add_relationship('laptimes', 'raceid', 'races', 'raceid')
    schema.add_relationship('laptimes', 'driverid', 'drivers', 'driverid')
    schema.add_relationship('pitstops', 'raceid', 'races', 'raceid')
    schema.add_relationship('pitstops', 'driverid', 'drivers', 'driverid')
    schema.add_relationship('qualifying', 'raceid', 'races', 'raceid')
    schema.add_relationship('qualifying', 'driverid', 'drivers', 'driverid')
    schema.add_relationship('qualifying', 'constructorid', 'constructors', 'constructorid')
    schema.add_relationship('results', 'raceid', 'races', 'raceid')
    schema.add_relationship('results', 'driverid', 'drivers', 'driverid')
    schema.add_relationship('results', 'constructorid', 'constructors', 'constructorid')
    schema.add_relationship('results', 'statusid', 'status', 'statusid')
    schema.add_relationship('target', 'raceid', 'races', 'raceid')
    schema.add_relationship('target', 'driverid', 'drivers', 'driverid')

    return schema
