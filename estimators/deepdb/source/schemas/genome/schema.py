from ensemble_compilation.graph_representation import SchemaGraph, Table


def gen_genome_schema(csv_path):
    """
    Schema for genome to do our experiment.
    """

    schema = SchemaGraph()

    # tables

    # att_classes
    schema.add_table(Table('att_classes', attributes=['att_class_id', 'att_class'],
                           primary_key=['att_class_id'],
                           irrelevant_attributes=['att_class'],
                           no_compression=[],
                           csv_file_location=csv_path.format('att_classes'),
                           table_size=699))
    
    # img_obj
    schema.add_table(Table('img_obj', attributes=['img_id', 'obj_sample_id', 'obj_class_id', 'x', 'y', 'w', 'h'],
                           primary_key=['img_id', 'obj_sample_id'],
                           irrelevant_attributes=[],
                           no_compression=[],
                           csv_file_location=csv_path.format('img_obj'),
                           table_size=1750617))

    # img_obj_att
    schema.add_table(Table('img_obj_att', attributes=['img_id', 'att_class_id', 'obj_sample_id'],
                           primary_key=['img_id', 'att_class_id', 'obj_sample_id'],
                           irrelevant_attributes=[],
                           no_compression=[],
                           csv_file_location=csv_path.format('img_obj_att'),
                           table_size=1074676))
    
    # img_rel
    schema.add_table(Table('img_rel', attributes=['img_id', 'pred_class_id', 'obj1_sample_id', 'obj2_sample_id'],
                           primary_key=['img_id', 'pred_class_id', 'obj1_sample_id', 'obj2_sample_id'],
                           irrelevant_attributes=[],
                           no_compression=[],
                           csv_file_location=csv_path.format('img_rel'),
                           table_size=763159))
    
    # obj_classes
    schema.add_table(Table('obj_classes', attributes=['obj_class_id', 'obj_class'],
                           primary_key=['obj_class_id'],
                           irrelevant_attributes=['obj_class'],
                           no_compression=[],
                           csv_file_location=csv_path.format('obj_classes'),
                           table_size=300))
    
    # pred_classes
    schema.add_table(Table('pred_classes', attributes=['pred_class_id', 'pred_class'],
                           primary_key=['pred_class_id'],
                           irrelevant_attributes=['pred_class'],
                           no_compression=['pred_class_id'],
                           csv_file_location=csv_path.format('pred_classes'),
                           table_size=150))
    
    schema.add_relationship('img_obj_att', 'att_class_id', 'att_classes', 'att_class_id')
    schema.add_relationship('img_obj_att', 'img_id', 'img_obj', 'img_id')
    schema.add_relationship('img_rel', 'img_id', 'img_obj', 'img_id')
    schema.add_relationship('img_obj', 'obj_class_id', 'obj_classes', 'obj_class_id')
    schema.add_relationship('img_rel', 'pred_class_id', 'pred_classes', 'pred_class_id')

    return schema
