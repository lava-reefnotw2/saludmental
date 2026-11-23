"""Seven real-data phases. Partial runs are never published as active results."""
from pathlib import Path
import sys,os,argparse,json,traceback,subprocess
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from training_jobs import BASE,atomic_json,preflight

def run(run_id,include_yrbs=False):
    out=BASE/'runs'/run_id
    os.environ['SP07_TRAINING_OUT']=str(out)
    import real_data_training as rt
    from threadpoolctl import threadpool_limits
    state=dict(run_id=run_id,state='running',include_yrbs=include_yrbs)
    def phase(number,message):
        state.update(phase=number,message=message,updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(out/'status.json',state)
        print(f'FASE {number}/7: {message}',flush=True)
    try:
        with threadpool_limits(limits=4):
            phase(1,'Ingesta: comprobando los ocho CSV y sus diccionarios locales')
            preflight(include_yrbs)
            phase(2,'Calidad: dominios, edades, etiquetas, pesos y diseño de encuesta')
            d=rt.prepare()
            phase(3,'Variables: núcleo común y bloque familiar; exclusión de etiquetas de X')
            forbidden={'y1_ideation','y2_plan','y3_attempt','y4_loneliness','y5_worry_sleep','country_iso3','weight','psu','stratum'}
            if set(rt.FULL)&forbidden:raise ValueError('Fuga en predictores')
            atomic_json(out/'variables.json',dict(core=rt.CORE,familiar=rt.FULL,target='y1_ideation',age=[13,14,15],structural_missing_country='GTM'))
            phase(4,'EDA: conteos y prevalencias observadas por país')
            rows=[]
            for country,g in d.groupby('country_iso3'):
                rows.append(dict(country=country,n=len(g),events=int(g.y1_ideation.sum()),prevalence_unweighted=float(g.y1_ideation.mean()),prevalence_weighted=float((g.y1_ideation*g.weight).sum()/g.weight.sum())))
            rt.pd.DataFrame(rows).to_csv(out/'eda_por_pais.csv',index=False)
            phase(5,'Entrenamiento: núcleo común de ocho países y comparación familiar controlada de cinco')
            rt.main(d=d)
            phase(6,'Calibración, bloque familiar de siete países y comparaciones pareadas')
            sys.path.insert(0,str(ROOT/'tools'))
            import complete_real_training
            complete_real_training.main()
            if include_yrbs:
                phase(6,'Réplica independiente YRBS: desarrollo 2021, calibración 2023 y prueba 2025')
                import train_yrbs_temporal
                train_yrbs_temporal.main()
            phase(7,'Comparaciones de ocho países, verificación e informe')
            from protocol_analysis import analyse, analyse_core
            from protocol_views import exports, provenance
            analyse_core(out)
            protocol_folder=analyse(out)
            from completar_auditoria import extend
            from protocol_explanations import elaborate
            extend(out)
            elaborate(out)
            from protocol_sensitivity import supplemental
            supplemental(out)
            exports(out,protocol_folder)
            atomic_json(protocol_folder/'procedencia.json',provenance(out))
            checks=json.loads((out/'verificaciones.json').read_text())
            if not checks.get('serialized_model_roundtrip'):raise ValueError('No se verificó el modelo serializado')
            lines=['# Entrenamiento real P07',f'Ejecución: {run_id}',
                   'Fuente: ocho CSV nacionales GSHS de datasets/. Edad: 13–15. Desenlace: ideación autorreportada en 12 meses.',
                   'Validación externa por país, selección interna agrupada, calibrador y umbral ajustados internamente. Las métricas son macro por país y ponderadas dentro del país.',
                   'Los gráficos del núcleo de ocho países usan protocolo/nucleo_8/. Las comparaciones familiares controladas de cinco países se guardan por separado en protocolo/. Optuna se configura desde la interfaz. Los intervalos son condicionales a países y modelos fijos, con 500 remuestreos de PSU dentro de estrato.',
                   'Resultados de investigación de asociación concurrente; no constituyen diagnóstico ni pronóstico individual.']
            for name in ['gshs5_core','gshs5_familiar','gshs8_core','gshs7_familiar']:
                lines += ['## '+name,'```csv',(out/name/'resumen_macro.csv').read_text(),'```']
            lines += ['## Comparaciones pareadas','```csv',(out/'comparaciones_pareadas.csv').read_text(),'```']
            (out/'INFORME_ENTRENAMIENTO.md').write_text('\n\n'.join(lines),encoding='utf-8')
            atomic_json(out/'run_manifest.json',dict(run_id=run_id,kind='empirical',include_yrbs=include_yrbs,n=len(d),events=int(d.y1_ideation.sum()),finished_at=datetime.now(timezone.utc).isoformat()))
            atomic_json(BASE/'active.json',dict(run_id=run_id))
            state.update(state='completed',phase=7,message='Entrenamiento real completado y resultados disponibles')
            atomic_json(out/'status.json',state)
    except Exception as exc:
        state.update(state='failed',message=str(exc))
        atomic_json(out/'status.json',state)
        traceback.print_exc()
        raise
    finally:(BASE/'training.lock').unlink(missing_ok=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--yrbs',action='store_true')
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or '/' in args.run_id or '\\' in args.run_id:raise ValueError('run_id inválido')
    run(args.run_id,args.yrbs)



