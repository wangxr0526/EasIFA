import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), '..')))
from werkzeug.utils import secure_filename
from wtforms.validators import DataRequired, Length, Email, EqualTo, NumberRange, regexp
from wtforms.fields import (StringField, PasswordField, DateField, BooleanField,
                            SelectField, SelectMultipleField, TextAreaField,
                            RadioField, IntegerField, DecimalField, SubmitField, FileField)
from wtforms import Form, validators
from flask_wtf import FlaskForm
from flask import Flask, render_template, request, redirect, url_for
from rdkit.Chem import AllChem
from flask_bootstrap import Bootstrap

from utils import EasIFAInferenceAPI, UniProtParser, UniProtParserMysql, get_structure_html_and_active_data, white_pdb,  reaction2svg, file_cache_path, rxn_fig_path, cmd, full_swissprot_checkpoint_path

res_colors={
    0: '#73B1FF',   # 非活性位点
    1: '#FF0000',     # Binding Site
    2: '#00B050',     # Active Site (Catalytic Site)
    3: '#FFFF00',     # Other Site
},

app = Flask(__name__)
WTF_CSRF_ENABLED = True  # prevents CSRF attacks
app.config['SECRET_KEY'] = 'abc123'


@app.before_first_request
def first_request():
    if os.path.exists(full_swissprot_checkpoint_path):
        app.ECSitePred = EasIFAInferenceAPI(model_checkpoint_path=full_swissprot_checkpoint_path)
    else:
        app.ECSitePred = EasIFAInferenceAPI()
    app.unprot_mysql_parser = UniProtParserMysql(mysql_config_path='./mysql_config.json')



class EnzymeRXNForm(Form):
    drawn_smiles = StringField(label='drawn_smiles')
    smiles = TextAreaField(label='smiles')
    file = FileField(label='Chose File',)
    uniprot_id = TextAreaField(label='uniprot_id')
    submit = SubmitField('Submit')
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/from_structure')
def from_structure():
    form = EnzymeRXNForm(request.form, request.files)
    return render_template('from_structure.html', form=form)

@app.route('/from_uniprot')
def from_uniprot():
    form = EnzymeRXNForm(request.form, request.files)
    return render_template('from_uniprot.html', form=form)


@app.route('/model_info')
def model_information():
    return render_template('model_information.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


# @app.route('/')
# def index():
#     form = EnzymeRXNForm(request.form)
#     return render_template('main/index.html', form=form)

# @app.route('/from_uniprot')
# def from_uniprot_submit():
#     # curl -H "Accept: text/plain; format=tsv" "https://rest.uniprot.org/uniprotkb/search?query=accession:Q7Z2W2&fields=accession,ec,sequence,cc_catalytic_activity,xref_alphafolddb"
#     form = EnzymeRXNForm(request.form)
#     return render_template('main/uniprot_input.html', form=form)
@app.template_filter('zip')
def zip_lists(a, b):
    return zip(a, b)
    

@app.route('/enzyme_structure', methods=['GET', 'POST'])
def load_enzyme_structure():
    
    if request.method == 'POST':
        form = EnzymeRXNForm(request.form, request.files)
        if request.files['file']:   
            pdb_lines = []
            if request.files.get('file').filename.split('.')[-1] == 'pdb':
                for line in request.files.get('file'):
                    print(line.decode('utf-8').strip())
                    pdb_lines.append(line.decode('utf-8').strip())      
            white_pdb(pdb_lines=pdb_lines)
    else:
        # form = MolForm(request.form)
        # return render_template('works/works_gostar_results.html', form=form)
        return redirect(url_for('index'))
    
    try:
        structure_html, _ = get_structure_html_and_active_data(enzyme_structure_path=os.path.join(file_cache_path, 'input_pdb.pdb'), view_size=(500, 440))
        
        return render_template('structure_template.html', ret={
            'error': None,
            'structure_html': structure_html,
        })
        
        
    except Exception as e:
        print(e)
        return render_template("main/results.html", ret={
            # 'form': form,
            'error': 'Input is not valid!',
            'output': [],
            # 'csv_id': csv_id,
        })

@app.route('/results_from_structure', methods=['GET', 'POST'])
def results_from_structure():
    
    all_list = []
    rxn_smiles = ''
    if request.method == 'POST':

        form = EnzymeRXNForm(request.form, request.files)

        print(form.drawn_smiles.data, form.file.data,
              form.smiles.data)

        if form.drawn_smiles.data:
            rxn_smiles = form.drawn_smiles.data

        if request.files['file']:
            pdb_lines = []
            if request.files.get('file').filename.split('.')[-1] == 'pdb':
                for line in request.files.get('file'):
                    print(line.decode('utf-8').strip())
                    pdb_lines.append(line.decode('utf-8').strip())      
            white_pdb(pdb_lines=pdb_lines)
            

        print(rxn_smiles)
    else:
        # form = MolForm(request.form)
        # return render_template('works/works_gostar_results.html', form=form)
        return redirect(url_for('index'))
    
    try:
        
        assert (rxn_smiles != '') and (len(pdb_lines) != 0)
        enzyme_structure_path=os.path.join(file_cache_path, 'input_pdb.pdb')
        pred_active_site_labels = app.ECSitePred.inference(rxn=rxn_smiles, enzyme_structure_path=enzyme_structure_path)
        if pred_active_site_labels is None:
            return  render_template("results_from_structure.html", ret={
                        # 'form': form,
                        'error': f'The enzyme is too large; enzymes with an amino acid count of {app.ECSitePred.max_enzyme_aa_length} are currently not supported for prediction.',
                        'output': [],
                        # 'csv_id': csv_id,
                    })
        enzyme_sequence = app.ECSitePred.caculated_sequence
        del app.ECSitePred.caculated_sequence
    
        structure_html, active_data = get_structure_html_and_active_data(enzyme_structure_path, site_labels=pred_active_site_labels, view_size=(600, 600))
        
        enzyme_sequence_color = ['#000000' for _ in range(len(enzyme_sequence))]
        for (idx_1_base, _, color, _) in active_data:
            enzyme_sequence_color[idx_1_base-1] = color
        
        grouped_enzyme_sequence = [enzyme_sequence[i:i+10] for i in range(0, len(enzyme_sequence), 10)]
        grouped_enzyme_sequence_index = [i+10 for i in range(0, len(enzyme_sequence), 10)]
        grouped_enzyme_sequence_color = [enzyme_sequence_color[i:i+10] for i in range(0, len(enzyme_sequence_color), 10)]
        if len(enzyme_sequence) % 10 != 0:
            grouped_enzyme_sequence_index[-1] = None
            grouped_enzyme_sequence[-1] = grouped_enzyme_sequence[-1] + ''.join(['&' for _ in range(10-len(grouped_enzyme_sequence[-1]))])
            grouped_enzyme_sequence_color[-1] = grouped_enzyme_sequence_color[-1] + ['#000000' for _ in range(10-len(grouped_enzyme_sequence_color[-1]))]
        
        rxnfigure_path = os.path.join(rxn_fig_path, 'rxn.svg')
        rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
        reaction2svg(rxn, rxnfigure_path)

        return render_template('results_from_structure.html', ret={
            'error': None,
            'structure_html': structure_html,
            # 'custom_css':custom_css,
            'rxnfigure_name': 'rxn.svg',
            'active_data': active_data,
            'enzyme_sequence_info': list(zip(grouped_enzyme_sequence, grouped_enzyme_sequence_index, grouped_enzyme_sequence_color))
        })
        
    except Exception as e:
        print(e)
        return render_template("results_from_structure.html", ret={
            # 'form': form,
            'error': 'Input is not valid!',
            'output': [],
            # 'csv_id': csv_id,
        })
        
    
# @app.route('/results', methods=['GET', 'POST'])
# def results():
    
#     all_list = []
#     rxn_smiles = ''
#     if request.method == 'POST':

#         form = EnzymeRXNForm(request.form, request.files)

#         print(form.drawn_smiles.data, form.file.data,
#               form.smiles.data)

#         if form.drawn_smiles.data:
#             rxn_smiles = form.drawn_smiles.data

#         if request.files['file']:
#             pdb_lines = []
#             if request.files.get('file').filename.split('.')[-1] == 'pdb':
#                 for line in request.files.get('file'):
#                     print(line.decode('utf-8').strip())
#                     pdb_lines.append(line.decode('utf-8').strip())      
#             white_pdb(pdb_lines=pdb_lines)
            

#         print(rxn_smiles)
#     else:
#         # form = MolForm(request.form)
#         # return render_template('works/works_gostar_results.html', form=form)
#         return redirect(url_for('index'))
    
#     try:
        
#         assert (rxn_smiles != '') and (len(pdb_lines) != 0)
#         enzyme_structure_path=os.path.join(file_cache_path, 'input_pdb.pdb')
#         pred_active_site_labels = app.ECSitePred.inference(rxn=rxn_smiles, enzyme_structure_path=enzyme_structure_path)
#         structure_html, active_data = get_structure_html_and_active_data(enzyme_structure_path, site_labels=pred_active_site_labels, view_size=(600, 600))
        
#         rxnfigure_path = os.path.join(rxn_fig_path, 'rxn.svg')
#         rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
#         reaction2svg(rxn, rxnfigure_path)

#         return render_template('main/results.html', ret={
#             'error': None,
#             'structure_html': structure_html,
#             # 'custom_css':custom_css,
#             'rxnfigure_name': 'rxn.svg',
#             'active_data': active_data,
#         })
        
#     except Exception as e:
#         print(e)
#         return render_template("main/results.html", ret={
#             # 'form': form,
#             'error': 'Input is not valid!',
#             'output': [],
#             # 'csv_id': csv_id,
#         })
        
# @app.route('/results_from_uniprot', methods=['GET', 'POST'])
# def results_from_uniprot():
    
#     if request.method == 'POST':

#         form = EnzymeRXNForm(request.form, request.files)

#         if form.uniprot_id.data:
#             uniprot_id = form.uniprot_id.data
#             print(form.uniprot_id.data)
                

#             print(uniprot_id)
#     else:
#         # form = MolForm(request.form)
#         # return render_template('works/works_gostar_results.html', form=form)
#         return redirect(url_for('index'))
    
#     try:
        
#         query_results_df = app.unprot_parser.parse_from_uniprotkb_query(uniprot_id)
        
#         structure_htmls = []
#         rxnfigure_names = []
#         active_data_list = []
  
#         for idx, row in enumerate(query_results_df.itertuples()):
#             rxn = row[2]
#             enzyme_structure_path = row[3]
#             pred_active_site_labels = app.ECSitePred.inference(rxn=rxn, enzyme_structure_path=enzyme_structure_path)
#             structure_html, active_data = get_structure_html_and_active_data(enzyme_structure_path=enzyme_structure_path, site_labels=pred_active_site_labels, view_size=(600, 600))
        
#             rxnfigure_path = os.path.join(rxn_fig_path, f'{uniprot_id}_rxn_{idx}.svg')
#             rxn = AllChem.ReactionFromSmarts(rxn, useSmiles=True)
#             reaction2svg(rxn, rxnfigure_path)
            
#             structure_htmls.append(structure_html)
#             rxnfigure_names.append(f'{uniprot_id}_rxn_{idx}.svg')
#             active_data_list.append(active_data)
            
#         _results = list(zip(list(range(len(structure_htmls))), structure_htmls, rxnfigure_names, active_data_list))

#         return render_template('main/results_from_uniprot.html', ret={
#             'error': None,
#             'results': _results,
#         })
        
#     except Exception as e:
#         print(e)
#         return render_template("main/results_from_uniprot.html", ret={
#             # 'form': form,
#             'error': 'Input is not valid!',
#             'output': [],
#             # 'csv_id': csv_id,
#         })

@app.route('/results_from_uniprot', methods=['GET', 'POST'])
def results_from_uniprot():
    
    if request.method == 'POST':

        form = EnzymeRXNForm(request.form, request.files)

        if form.uniprot_id.data:
            uniprot_id = form.uniprot_id.data.strip().upper()
            print(form.uniprot_id.data)
                

            print(uniprot_id)
    else:
        # form = MolForm(request.form)
        # return render_template('works/works_gostar_results.html', form=form)
        return redirect(url_for('index'))
    
    try:
        use_store_results = False
        query_data, stored_predicted_results, is_new_data = app.unprot_mysql_parser.query_from_uniprot(uniprot_id)
        uniprot_id, query_results_df, msg, stored_calculated_sequence = query_data
        
        # query_results_df, msg = app.unprot_parser.parse_from_uniprotkb_query(uniprot_id)

        if (msg != 'Good') and is_new_data:
            predicted_results = []
            app.unprot_mysql_parser.insert_to_local_database(uniprot_id=uniprot_id, query_dataframe=query_results_df, message=msg, calculated_sequence=stored_calculated_sequence, predicted_results=predicted_results)
        
        if msg == 'Not Enzyme':
            return  render_template("results_from_structure.html", ret={
                            # 'form': form,
                            'error': f'The UniProt ID: {uniprot_id} appears not to be an enzyme.',
                            'output': [],
                            # 'csv_id': csv_id,
                        })
        if msg == 'No recorded reaction catalyzed found':
            return  render_template("results_from_structure.html", ret={
                            # 'form': form,
                            'error': f'No reactions catalyzed were found for Uniprot ID: {uniprot_id}',
                            'output': [],
                            # 'csv_id': csv_id,
                        })
        
        if msg == 'No Alphafolddb Structure':
            return  render_template("results_from_structure.html", ret={
                            # 'form': form,
                            'error': f'Uniprot ID: {uniprot_id} could not find available structural data',
                            'output': [],
                            # 'csv_id': csv_id,
                        })
        
        enzyme_aa_length = query_results_df['aa_length'].tolist()[0]
        if enzyme_aa_length > app.ECSitePred.max_enzyme_aa_length:
            return  render_template("results_from_structure.html", ret={
                            # 'form': form,
                            'error': f'The enzyme is too large; enzymes with an amino acid count of {app.ECSitePred.max_enzyme_aa_length} are currently not supported for prediction.',
                            'output': [],
                            # 'csv_id': csv_id,
                        })
            
            
        if stored_predicted_results and len(stored_predicted_results) == len(query_results_df):
            use_store_results = True
            
        
        structure_htmls = []
        rxnfigure_names = []
        active_data_list = []
        enzyme_sequence_info = []
        predicted_results = []
        calculated_sequence = None
  
        for idx, row in enumerate(query_results_df.itertuples()):
            rxn = row[2]
            enzyme_structure_path = row[3]
            if not os.path.exists(enzyme_structure_path):
                enzyme_structure_path = os.path.join(app.unprot_mysql_parser.unprot_parser.alphafolddb_folder, f'AF-{uniprot_id}-F1-model_v4.pdb')
                cmd(app.unprot_mysql_parser.unprot_parser.download_alphafolddb_url_template.format(enzyme_structure_path, uniprot_id))
            
            
            if use_store_results:
                pred_active_site_labels = stored_predicted_results[idx]
            else:
                pred_active_site_labels = app.ECSitePred.inference(rxn=rxn, enzyme_structure_path=enzyme_structure_path)
                predicted_results.append(pred_active_site_labels)
            if pred_active_site_labels is None:
                return  render_template("results_from_structure.html", ret={
                            # 'form': form,
                            'error': f'The enzyme is too large; enzymes with an amino acid count of {app.ECSitePred.max_enzyme_aa_length} are currently not supported for prediction.',
                            'output': [],
                            # 'csv_id': csv_id,
                        })
                
            structure_html, active_data = get_structure_html_and_active_data(enzyme_structure_path=enzyme_structure_path, site_labels=pred_active_site_labels, view_size=(600, 600))
            
            if use_store_results:
                enzyme_sequence = stored_calculated_sequence
            else: 
                enzyme_sequence = app.ECSitePred.caculated_sequence
                del app.ECSitePred.caculated_sequence
                if not calculated_sequence:
                    calculated_sequence = enzyme_sequence

            enzyme_sequence_color = ['#000000' for _ in range(len(enzyme_sequence))]
            for (idx_1_base, _, color, _) in active_data:
                enzyme_sequence_color[idx_1_base-1] = color
            
            grouped_enzyme_sequence = [enzyme_sequence[i:i+10] for i in range(0, len(enzyme_sequence), 10)]
            grouped_enzyme_sequence_index = [i+10 for i in range(0, len(enzyme_sequence), 10)]
            grouped_enzyme_sequence_color = [enzyme_sequence_color[i:i+10] for i in range(0, len(enzyme_sequence_color), 10)]
            if len(enzyme_sequence) % 10 != 0:
                grouped_enzyme_sequence_index[-1] = None
                grouped_enzyme_sequence[-1] = grouped_enzyme_sequence[-1] + ''.join(['&' for _ in range(10-len(grouped_enzyme_sequence[-1]))])
                grouped_enzyme_sequence_color[-1] = grouped_enzyme_sequence_color[-1] + ['#000000' for _ in range(10-len(grouped_enzyme_sequence_color[-1]))]
        
            rxnfigure_path = os.path.join(rxn_fig_path, f'{uniprot_id}_rxn_{idx}.svg')
            rxn = AllChem.ReactionFromSmarts(rxn, useSmiles=True)
            reaction2svg(rxn, rxnfigure_path)
            
            structure_htmls.append(structure_html)
            rxnfigure_names.append(f'{uniprot_id}_rxn_{idx}.svg')
            active_data_list.append(active_data)
            enzyme_sequence_info.append(list(zip(grouped_enzyme_sequence, grouped_enzyme_sequence_index, grouped_enzyme_sequence_color)))
            
        _results = list(zip(list(range(len(structure_htmls))), structure_htmls, rxnfigure_names, active_data_list, enzyme_sequence_info))
        
        if is_new_data:
            app.unprot_mysql_parser.insert_to_local_database(uniprot_id=uniprot_id, query_dataframe=query_results_df, message=msg, calculated_sequence=calculated_sequence, predicted_results=predicted_results)

        return render_template('results_from_uniprot.html', ret={
            'error': None,
            'results': _results,
        })
        
    except Exception as e:
        print(e)
        return render_template("results_from_structure.html", ret={
            # 'form': form,
            'error': 'The input is not valid, or there is an internal problem with the system. Please contact the administrator, Email: wangxr2018@lzu.edu.cn',
            'output': [],
            # 'csv_id': csv_id,
        })

if __name__ == '__main__':
    from pathlib import Path
    cur_file_path = Path(__file__).resolve().parent  # Path.cwd().parent   #
    app.config['UPLOAD_FOLDER'] = cur_file_path/'upload'
    app.config['MAX_CONTENT_PATH'] = 2**10
    Bootstrap(app)
    app.run(host='0.0.0.0', port=8000, debug=True)
