"""Local interactive application. INT8 is the default actual inference model."""
import json
import os
import time
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
import gradio as gr
from models import Engine, ROOT
from preprocessing import normalize, rule_predict

engine=Engine()

def moderate(message, model, history, logs):
    if not message or not message.strip():
        raise gr.Error('Enter a message first.')
    if len(message)>2000:
        raise gr.Error('Please keep messages under 2,000 characters.')
    start=time.perf_counter();prob=engine.probability(message,model);ms=(time.perf_counter()-start)*1000
    flagged=prob>=.5;label='OFFENSIVE' if flagged else 'CLEAN';confidence=prob if flagged else 1-prob
    shown='[Message hidden by moderation]' if flagged else message
    history=list(history or [])+[{'role':'user','content':shown},
        {'role':'assistant','content':f'{label} | score {confidence:.1%} | {model} | {ms:.2f} ms'}]
    short_model=dict(zip(engine.names,['TF-IDF','W2V + NN','BERT FP32','BERT INT8']))[model]
    logs=list(logs or [])+[[len(logs or [])+1,short_model,label,round(confidence,3),round(ms,2)]]
    return '',history,logs,logs,f'**{label}** · {confidence:.1%} model score · **{ms:.2f} ms** on CPU'

def compare(message):
    if not message.strip():raise gr.Error('Enter a comparison message.')
    rows=[]
    for name in engine.names:
        t=time.perf_counter();p=engine.probability(message,name);ms=(time.perf_counter()-t)*1000
        rows.append([name,'OFFENSIVE' if p>=.5 else 'CLEAN',round(p,4),round(ms,2)])
    return rows,normalize(message),('OFFENSIVE' if rule_predict(message) else 'CLEAN')

with gr.Blocks(title='ChatGuard | SWE015',analytics_enabled=False,theme=gr.themes.Soft(primary_hue='teal'),css='footer {display:none} .gradio-container {max-width:1200px!important}') as demo:
    gr.Markdown('# ChatGuard\n### Real-time chat moderation\nEnglish messages · CPU inference · BERT INT8 by default')
    with gr.Tab('Live chat'):
        with gr.Row():
            with gr.Column(scale=3):
                chat=gr.Chatbot(type='messages',height=360,label='Moderated conversation')
                message=gr.Textbox(label='New message',placeholder='Type a message and press Enter…',max_lines=3)
                send=gr.Button('Send message',variant='primary')
                status=gr.Markdown('Ready. Flagged messages are hidden from the conversation.')
            with gr.Column(scale=2):
                model=gr.Dropdown(engine.names,value='BERT INT8',label='Active model')
                gr.Markdown('**Latency budget:** 50 ms mean on CPU.\n\nScores are model estimates, not calibrated guarantees. Messages are evaluated independently; only the first 96 WordPiece tokens are used by BERT.')
                table=gr.Dataframe(headers=['#','Model','Label','Score','ms'],column_widths=['35px','100px','95px','65px','70px'],wrap=True,interactive=False,label='Response log')
                clear=gr.Button('Clear conversation')
        state=gr.State([])
        for event in (send.click,message.submit):
            event(moderate,[message,model,chat,state],[message,chat,table,state,status])
        clear.click(lambda: ([],[],[], 'Conversation cleared.'),outputs=[chat,table,state,status])
    with gr.Tab('Compare models'):
        sample=gr.Textbox(label='Same input for every model',value='Thanks for helping our team!')
        button=gr.Button('Compare all four models',variant='primary')
        comparison=gr.Dataframe(headers=['Model','Label','P(offensive)','Latency (ms)'],interactive=False)
        normalized=gr.Textbox(label='Normalized text',interactive=False)
        rule=gr.Textbox(label='Extra edit-distance rule baseline',interactive=False)
        button.click(compare,sample,[comparison,normalized,rule])
    with gr.Tab('Measured results'):
        path=ROOT/'results'/'metrics.json'
        if path.exists():
            results=json.loads(path.read_text(encoding='utf-8'))
            gr.Dataframe(value=[[r['model'],round(r['macro_f1'],4),round(r['size_mb'],2),round(r['latency_mean_ms'],2)] for r in results['models']],headers=['Model','Macro-F1','Size MB','Mean ms'],interactive=False)
            gr.Markdown(f"Held-out test messages: {results['test_n']}. 200 measured runs after 20 warm-ups, batch size 1, two CPU threads.\n\n{results['hardware']}\n\nLatency includes preprocessing and inference, excludes browser/network.")
    gr.Markdown('Ahmet Cihan Aycan · 220911804 · SWE015 · Use Case 4\n\nAcademic prototype trained on English social-media messages; gaming chat may differ.')

if __name__=='__main__':demo.launch(server_name='127.0.0.1',inbrowser=False)
