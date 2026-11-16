"""Aggregate-only final report for the completed historical factorial.

Does not read tracking, labels, frames or prediction rows. Prediction files are
hashed as opaque bytes solely to identify identical full outputs. Refuses to read
saved public aggregate scores until the freeze and completion markers exist.
All writes stay under the experiment's ignored reports/ directory.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DEST=ROOT/'runs/historical-factorial-20261116'
FACTORS=('boundary_weighting','augmentation','long_context','external_expansion')
METRICS=('accuracy_pct','macroF1','spellF1','dead_ball_recall')
SPELL_METRICS=('spell_precision_05','spell_recall_05','missed_possessions','extra_possessions')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def table(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def decoder_key(value):
    return json.dumps({k:v for k,v in value.items() if k!='id'},sort_keys=True)


def identity(row):
    return row['candidate'],decoder_key(row['decoder'])


def fixed_reference(row,weight):
    s=row['decoder']
    return (row['candidate']=='historical' and s['duration']=='hard' and s['minimum']==1.
            and s['transition_penalty']==0. and s['missing_weight']==weight)


def deltas(after,before):
    return {key:after[key]-before[key] for key in METRICS+SPELL_METRICS if key in after and key in before}


def factorial_effects(matrix,recipes):
    """Matched contrasts, keeping other factors/selector/decoder fixed.

    These are descriptive contrasts on shared validation matches, not independent
    statistical observations or estimates of public performance.
    """
    records={identity(row):row for row in recipes}
    index={tuple(arm[f] for f in FACTORS):arm for arm in matrix if arm['seed']==42}
    decoder_keys=sorted({decoder_key(row['decoder']) for row in recipes})
    effects=[]
    def collect(name,pairs):
        values=[]
        for before_name,after_name in pairs:
            for key in decoder_keys:
                before=records.get((before_name,key));after=records.get((after_name,key))
                if before and after:values.append(deltas(after['mean'],before['mean']))
        if values:
            effect=dict(factor=name,matched_recipe_pairs=len(values))
            for metric in METRICS+SPELL_METRICS:
                observed=[row[metric] for row in values if metric in row]
                if observed:effect['mean_delta_'+metric]=statistics.mean(observed)
            effect['positive_spellF1_pairs']=sum(row['spellF1']>0 for row in values)
            effects.append(effect)
    for i,factor in enumerate(FACTORS):
        pairs=[]
        for levels,off in index.items():
            if levels[i]:continue
            on_levels=list(levels);on_levels[i]=True
            on=index.get(tuple(on_levels))
            if on:
                pairs += [(off['id']+'_'+selector,on['id']+'_'+selector) for selector in ('raw_best','decoded_best')]
        collect(factor,pairs)
    collect('decoded_checkpoint_selection',[(arm['id']+'_raw_best',arm['id']+'_decoded_best') for arm in index.values()])
    return effects


def joint_diagnostic_ranking(public_rows,recipes,prediction_hashes=None):
    """Rank frozen public recipes subject to explicit external safeguards.

    This is a diagnostic presentation only, not a new adopted candidate. Filter
    individual recipes BEFORE identical-public-output grouping: two aliases may
    produce equal public labels while differing in external performance.
    """
    reference_public=next(row for row in public_rows if fixed_reference(row,.5))
    reference=next(row for row in recipes if fixed_reference(row,.5))
    lookup={identity(row):row for row in recipes}
    ordered=sorted(public_rows,key=lambda row:(-row['public']['spellF1'],
        -row['public']['macroF1'],row['candidate'],decoder_key(row['decoder'])))
    eligible=[]
    for row in ordered:
        external=lookup[identity(row)]
        if (external['mean']['spellF1']>reference['mean']['spellF1']+1e-12
                and external['mean']['missed_possessions']<reference['mean']['missed_possessions']-1e-12):
            eligible.append(row)
    groups={}
    for name,strict in [('external_gain_and_fewer_misses',False),('also_no_match_spellF1_regression',True)]:
        rows=[]
        for row in eligible:
            external=lookup[identity(row)]
            consistent=all(external['per_match'][mid]['spellF1']>=metrics['spellF1']-1e-12
                           for mid,metrics in reference['per_match'].items())
            if not strict or consistent:rows.append(row)
        groups[name]=dict(eligible_frozen_recipes=len(rows),
            distinct_full_predictions=len({prediction_hashes[row['name']] for row in rows}) if prediction_hashes is not None else None,
            best=None)
        if rows:
            row=rows[0];external=lookup[identity(row)]
            groups[name]['best']=dict(candidate=row['candidate'],decoder=row['decoder'],
                external=row['external'],public=row['public'],
                change_from_historical_weight05_external=deltas(external['mean'],reference['mean']),
                change_from_historical_weight05_public=deltas(row['public'],reference_public['public']),
                external_per_match_spellF1_changes={mid:external['per_match'][mid]['spellF1']-metrics['spellF1']
                    for mid,metrics in reference['per_match'].items()})
    return dict(reference=dict(candidate='historical',decoder=reference['decoder']),
        criteria='External mean spellF1 strictly improves and average missed possessions strictly decreases versus historical hard1s transition0 missing_weight0.5. Strict group additionally requires no per-match external spellF1 regression. Rank eligible frozen recipes by public spellF1 then public macroF1; report accuracy and extra-possession changes without hiding trade-offs.',
        diagnostic_only=True,does_not_change_external_selection=True,groups=groups)


def training_fairness_fields(status,arm,configuration):
    """Recover old base-run accounting without inventing completed measurements."""
    available=status.get('train_windows')
    cap=status.get('epoch_windows')
    cap_source='recorded status'
    if cap is None:
        cap=configuration.get('config',{}).get('epoch_windows',arm.get('epoch_windows'))
        cap_source='configured cap'
    if cap is None and not arm['external_expansion']:
        cap=available;cap_source='legacy base status train_windows'
    batch=configuration.get('training',{}).get('training',{}).get('batch_size')
    steps=status.get('optimizer_steps_per_epoch');steps_source='recorded status'
    if steps is None and cap is not None and batch:
        steps=(cap+batch-1)//batch;steps_source='derived ceil(epoch_windows/batch_size)'
    return dict(available_train_windows=available,epoch_windows=cap,epoch_windows_source=cap_source,
        batch_size=batch,optimizer_steps_per_epoch=steps,optimizer_steps_source=steps_source,
        epoch_sampling='not recorded' if cap is None else 'random subset without replacement' if arm['external_expansion'] else 'full shuffled window set')


def duplicate_scenario_sensitivity(recipes,selected):
    """Remove speed_missing for fixed frozen recipes; never reselect a model."""
    pairs=0;identical=0
    for recipe in recipes:
        per_case={(row['match_id'],row['scenario']):row for row in recipe['rows']}
        for match in recipe['per_match']:
            clean=per_case.get((match,'clean'));speed=per_case.get((match,'speed_missing'))
            if clean is not None and speed is not None:
                pairs+=1
                keys=METRICS+SPELL_METRICS+('gt_possessions','pred_possessions')
                if all(clean.get(key)==speed.get(key) for key in keys):identical+=1
    leaders=[('external_selected',selected)]+[(f'historical_weight_{weight}',next(row for row in recipes if fixed_reference(row,weight))) for weight in (0.,.5)]
    results=[]
    for role,recipe in leaders:
        cases=[row for row in recipe['rows'] if row['scenario']!='speed_missing']
        per_match={}
        for match in recipe['per_match']:
            subset=[row for row in cases if row['match_id']==match]
            per_match[match]={key:statistics.mean(row[key] for row in subset) for key in METRICS+SPELL_METRICS if subset and all(key in row for row in subset)}
        means={key:statistics.mean(row[key] for row in per_match.values()) for key in METRICS+SPELL_METRICS if per_match and all(key in row for row in per_match.values())}
        results.append(dict(role=role,candidate=recipe['candidate'],decoder=recipe['decoder'],
            retained_scenarios=sorted({row['scenario'] for row in cases}),mean=means,per_match=per_match,
            change_from_original_six_condition_mean=deltas(means,recipe['mean'])))
    changes={row['role']:deltas(results[0]['mean'],row['mean']) for row in results[1:]}
    return dict(diagnostic_only=True,reselection_performed=False,
        original_selection_unchanged=True,checked_clean_speed_missing_pairs=pairs,identical_metric_pairs=identical,
        all_checked_pairs_identical=bool(pairs) and pairs==identical,
        explanation='Recorded ball speed is always hidden. Clean and speed_missing are therefore the same condition: the six-condition selection objective contains five distinct perturbation conditions and gives clean a second weight. This table omits speed_missing only for the already frozen external-selected recipe and two historical anchors; no ranking search or reselection is performed.',
        recipes=results,selected_changes_from_five_condition_anchors=changes)


def run(dest):
    dest=Path(dest).resolve()
    if dest!=DEFAULT_DEST.resolve():
        # A synthetic fixture may live below runs, but never let reports write elsewhere.
        assert dest.is_relative_to(ROOT/'runs'),'Reports must stay under the ignored runs directory'
    required=[dest/'freeze.json',dest/'public/results.json',dest/'public/predictions_frozen.json',dest/'summary.json',dest/'verification.json']
    if not all(p.exists() for p in required):
        print('NOT READY: complete freeze, public aggregate results and completion markers are required; no public scores read.')
        return False
    freeze=read(dest/'freeze.json');verification=read(dest/'verification.json')
    assert verification.get('outputs_ignored') and verification.get('reference_unchanged'),'Pipeline verification incomplete'
    plan=read(dest/'protocol.json');matrix=read(dest/'training_matrix.json')
    recipes=freeze['recipes'];manifest=read(dest/'public/predictions_frozen.json')
    public=read(dest/'public/results.json')
    assert len(public)==len(manifest)==len(recipes),'Public results incomplete'
    assert {row['name'] for row in public}=={row['name'] for row in manifest}
    assert {identity(row) for row in public}=={identity(row) for row in recipes}
    lookup={identity(row):row for row in recipes}
    selected=next(row for row in public if identity(row)==identity(freeze['external_selected']))
    for record in freeze['preserved']+freeze.get('source_hashes',[]):
        assert sha(record['path'])==record['sha256'],'Frozen asset changed: '+record['path']
    checkpoints={row['checkpoint']['path']:row['checkpoint'] for row in recipes if 'checkpoint' in row}
    for record in checkpoints.values():
        assert sha(record['path'])==record['sha256'],'Checkpoint changed after freeze'

    training=[]
    configured={arm['id']:arm for arm in matrix}
    confirmation_path=dest/'confirmation_plan.json'
    if confirmation_path.exists():
        for arm in read(confirmation_path):
            second=dict(arm,seed=43,id=arm['id'][:-2]+'43');configured[second['id']]=second
    for name,arm in sorted(configured.items()):
        path=dest/'training_status'/(name+'.json')
        fallback=dest/'models'/name/'status.json'
        status=read(path) if path.exists() else read(fallback) if fallback.exists() else {'status':'NOT RUN'}
        config_path=dest/'models'/name/'config.json'
        configuration=read(config_path) if config_path.exists() else {}
        training.append(dict(id=name,seed=arm['seed'],**{k:arm[k] for k in FACTORS},
            **training_fairness_fields(status,arm,configuration),
            status=status['status'],epochs_run=status.get('epochs_run'),
            raw_best_epoch=status.get('best_epochs',{}).get('raw'),decoded_best_epoch=status.get('best_epochs',{}).get('decoded'),
            seconds=status.get('seconds'),parameters=status.get('parameters'),
            receptive_field_frames=status.get('receptive_field_frames'),stopping_reason=status.get('stopping_reason'),
            reason=status.get('error',status.get('reason'))))
    counts=dict(screening_planned=len(matrix),screening_completed=sum(row['status']=='COMPLETE' and row['seed']==42 for row in training),
        seed43_planned=sum(row['seed']==43 for row in training),seed43_completed=sum(row['status']=='COMPLETE' and row['seed']==43 for row in training),
        completed_training_runs=sum(row['status']=='COMPLETE' for row in training),
        incomplete_runs=[row['id'] for row in training if row['status']!='COMPLETE'],
        evaluated_candidate_ids=len({row['candidate'] for row in recipes}),evaluated_checkpoint_files=len(checkpoints),
        decoder_settings=len({decoder_key(row['decoder']) for row in recipes}),frozen_recipes=len(recipes),public_scored_recipes=len(public))
    base_runs=[r for r in training if r['status']=='COMPLETE' and not r['external_expansion']]
    expanded_runs=[r for r in training if r['status']=='COMPLETE' and r['external_expansion']]
    base_caps=sorted({r['epoch_windows'] for r in base_runs if r['epoch_windows'] is not None})
    expanded_caps=sorted({r['epoch_windows'] for r in expanded_runs if r['epoch_windows'] is not None})
    base_steps=sorted({r['optimizer_steps_per_epoch'] for r in base_runs if r['optimizer_steps_per_epoch'] is not None})
    expanded_steps=sorted({r['optimizer_steps_per_epoch'] for r in expanded_runs if r['optimizer_steps_per_epoch'] is not None})
    amendment_path=dest/'expansion_compute_control.json'
    compute_control=dict(amendment=read(amendment_path) if amendment_path.exists() else None,
        base_epoch_windows=base_caps,expanded_epoch_windows=expanded_caps,
        base_optimizer_steps_per_epoch=base_steps,expanded_optimizer_steps_per_epoch=expanded_steps,
        epoch_windows_matched=(len(base_caps)==1 and expanded_caps==base_caps) if expanded_runs else None,
        optimizer_steps_matched=(len(base_steps)==1 and expanded_steps==base_steps) if expanded_runs else None,
        policy='Expanded arms sample a random subset without replacement each epoch, capped to the base window count. Optimizer updates per epoch match; selected epoch counts and numbers of supervised frames can differ because weak labels are partial.')
    out=dest/'reports';out.mkdir(exist_ok=True)
    table(out/'training_runs.csv',training)
    external_summary=[];external_matches=[];external_conditions=[]
    for row in recipes:
        base=dict(candidate=row['candidate'],decoder=row['decoder'].get('id',decoder_key(row['decoder'])))
        external_summary.append(dict(**base,**row['mean']))
        external_matches += [dict(**base,match_id=mid,**values) for mid,values in row['per_match'].items()]
        external_conditions += [dict(**base,**{k:v for k,v in values.items() if k not in ('candidate','decoder')}) for values in row['rows']]
    external_summary.sort(key=lambda row:(-row['spellF1'],-row['macroF1']))
    table(out/'external_all_recipes.csv',external_summary)
    table(out/'external_per_match.csv',external_matches)
    table(out/'external_per_match_scenario.csv',external_conditions)
    effects=factorial_effects(matrix,recipes)
    table(out/'factorial_effects.csv',effects)

    # Ranking is presentation only; externally selected candidate remains frozen.
    ordered=sorted(public,key=lambda row:(-row['public']['spellF1'],-row['public']['macroF1'],row['candidate'],decoder_key(row['decoder'])))
    unique=[];by_hash={};prediction_hashes={}
    for row in ordered:
        digest=sha(row['path'])
        prediction_hashes[row['name']]=digest
        alias=dict(candidate=row['candidate'],decoder=row['decoder'].get('id'),name=row['name'])
        if digest in by_hash:by_hash[digest]['equivalent_full_predictions'].append(alias)
        else:
            entry=dict(row,prediction_sha256=digest,equivalent_full_predictions=[])
            by_hash[digest]=entry;unique.append(entry)
    counts['distinct_full_prediction_files']=len(unique)
    counts['identical_prediction_aliases']=len(public)-len(unique)
    write(out/'public_unique_predictions.json',unique)
    table(out/'public_unique_predictions.csv',[dict(rank=i+1,candidate=row['candidate'],decoder=row['decoder'].get('id'),
        external_spellF1=row['external']['spellF1'],prediction_sha256=row['prediction_sha256'],
        identical_alias_count=len(row['equivalent_full_predictions']),**row['public']) for i,row in enumerate(unique)])
    joint=joint_diagnostic_ranking(public,recipes,prediction_hashes)
    write(out/'joint_external_public_diagnostics.json',joint)
    joint_table=[]
    for name,group in joint['groups'].items():
        row=dict(group=name,eligible_frozen_recipes=group['eligible_frozen_recipes'],distinct_full_predictions=group['distinct_full_predictions'])
        best=group['best']
        if best:
            row.update(candidate=best['candidate'],decoder=best['decoder'].get('id'))
            for domain in ('external','public'):
                row.update({domain+'_'+k:v for k,v in best[domain].items() if k in METRICS+SPELL_METRICS})
                row.update({domain+'_delta_'+k:v for k,v in best['change_from_historical_weight05_'+domain].items()})
        else:row['candidate']='NO QUALIFYING RECIPE'
        joint_table.append(row)
    table(out/'joint_external_public_diagnostics.csv',joint_table)
    references=[]
    for weight in (0.,.5):
        before=next(row for row in public if fixed_reference(row,weight))
        references.append(dict(missing_weight=weight,candidate=before['candidate'],decoder=before['decoder'],
            external=before['external'],public=before['public'],
            external_selected_change= dict(external=deltas(selected['external'],before['external']),public=deltas(selected['public'],before['public'])),
            public_winner_change=dict(external=deltas(unique[0]['external'],before['external']),public=deltas(unique[0]['public'],before['public']))))
    selected_external=lookup[identity(selected)]
    sensitivity=duplicate_scenario_sensitivity(recipes,selected_external)
    write(out/'five_condition_sensitivity.json',sensitivity)
    sensitivity_rows=[]
    for row in sensitivity['recipes']:
        base=dict(role=row['role'],candidate=row['candidate'],decoder=row['decoder'].get('id'))
        sensitivity_rows.append(dict(**base,scope='equal_match_mean',**row['mean']))
        sensitivity_rows += [dict(**base,scope=mid,**values) for mid,values in row['per_match'].items()]
    table(out/'five_condition_sensitivity.csv',sensitivity_rows)
    per_match_changes=[]
    for weight in (0.,.5):
        before=next(row for row in recipes if fixed_reference(row,weight))
        for mid,values in selected_external['per_match'].items():
            per_match_changes.append(dict(reference_weight=weight,match_id=mid,**deltas(values,before['per_match'][mid])))
    table(out/'external_selected_per_match_changes.csv',per_match_changes)

    heldout_path=dest/'untouched_test_confirmation.json'
    heldout=read(heldout_path) if heldout_path.exists() else dict(rows=[],regime='NOT RUN')
    table(out/'heldout_confirmation.csv',heldout['rows'])
    heldout_changes=[]
    for row in heldout['rows']:
        if row['candidate']!=selected['candidate']:continue
        for before in heldout['rows']:
            if before['match_id']==row['match_id'] and before['candidate'].startswith('historical_weight_'):
                heldout_changes.append(dict(match_id=row['match_id'],reference=before['candidate'],**deltas(row,before)))
    expansion_path=dest/'external_expansion/manifest.json'
    expansion=read(expansion_path) if expansion_path.exists() else {}
    recovery_path=dest/'io_recovery_audit.json'
    recovery=read(recovery_path) if recovery_path.exists() else {}
    synthetic_recovery_path=ROOT/'runs/historical-factorial-recovery-synthetic-tests-20261116/synthetic_recovery_test.json'
    operational_recovery=dict(semantics=freeze.get('recovery',recovery.get('semantics','No recovery metadata in freeze')),
        retry_events=len(recovery.get('retries',[])),recovery_audit=str(recovery_path) if recovery_path.exists() else None,
        training_recovery_manifest=str(dest/'file_lock_recovery.json') if (dest/'file_lock_recovery.json').exists() else None,
        synthetic_recovery_test=read(synthetic_recovery_path) if synthetic_recovery_path.exists() else None,
        note='Recovery reuses frozen recipes and complete heldout/public aggregates. It does not alter model selection or tune from scores. Synthetic tests are separate from actual pipeline completion verification.')
    split=plan['splits']
    limits=[
        'External development scores reuse a small validation set across many recipes; they are not unbiased test estimates.',
        'Clean and speed_missing are exact duplicate conditions for the speed-masked historical feature pipeline. The original six-condition objective double-weights clean and has five unique perturbation conditions; the fixed-recipe five-condition sensitivity is diagnostic only.',
        'Public scores use the repeatedly inspected five-minute window. Public winners are diagnostic rankings, not evidence of hidden-match generalisation.',
        'The best observed public score is not guaranteed to repeat, and reaching0.800 is not guaranteed.',
        'Identical full prediction CSVs are grouped as aliases; distinct files may still tie or produce identical labels inside the public window.',
        'Factorial effects are matched external contrasts averaged over declared decoders and checkpoint selectors; they are descriptive, not independent statistical tests.',
        'External expansion adds partial provider-derived H/A supervision only. It supplies no independently labelled dead-ball examples and is not equivalent to two fully labelled H/A/D matches.',
        'Matching epoch windows and optimizer updates does not equalise labelled-frame counts: partial weak labels can leave different numbers of supervised frames in sampled windows. Early stopping can also produce different total epoch/update counts.',
        'Held-out confirmation is clean-only, run after freezing selection; its results must not change the selected checkpoint or decoder.',
    ]
    if counts['incomplete_runs']:limits.append('Incomplete planned runs: '+', '.join(counts['incomplete_runs']))
    if not heldout['rows']:limits.append('Untouched external-test confirmation was not completed.')
    if expanded_runs and not compute_control['optimizer_steps_matched']:
        limits.append('Expanded/base optimizer updates per epoch were not verified equal; inspect training_runs.csv and compute_control in summary.json.')
    summary=dict(counts=counts,training=training,references=references,
        compute_control=compute_control,duplicate_scenario_sensitivity=sensitivity,operational_recovery=operational_recovery,
        external_selected=dict(candidate=selected['candidate'],decoder=selected['decoder'],external=selected['external'],public=selected['public'],
            per_match=selected_external['per_match'],per_match_changes=per_match_changes,rationale=freeze['reason']),
        public_top_three=unique[:3],joint_external_public_diagnostics=joint,factorial_effects=effects,
        heldout_confirmation=heldout,heldout_changes=heldout_changes,
        data=dict(strong_training_matches=len(split['train']),validation_matches=len(split['validation']),reserved_test_matches=len(split['test']),
            actual_new_matches=expansion.get('actual_new_matches',0),weak_supervision=expansion.get('weak_supervision',False),
            expansion_has_full_had_ground_truth=expansion.get('full_had_ground_truth',False),expansion_match_ids=expansion.get('match_ids',[])),
        limitations=limits,active_submission_unchanged=True,selection_changed_from_public=False)
    write(out/'summary.json',summary)
    commands=(dest/'commands.txt').read_text() if (dest/'commands.txt').exists() else ''
    commands+='python experiments/boundary_motion/historical_factorial_report.py\n'
    (out/'commands.txt').write_text(commands,encoding='utf-8')
    text=['Historical factorial: completed aggregate report',
        'Training, checkpoint selection and decoder tuning used external data. All public recipes were frozen before the diagnostic batch.',
        'Existing submission and original historical checkpoints remain unchanged.',
        'Execution coverage: '+json.dumps(counts),
        'Expansion compute fairness (actual recorded values; old base runs use train_windows fallback): '+json.dumps(compute_control),
        f"Data: {len(split['train'])} strongly labelled training matches; {expansion.get('actual_new_matches',0)} extra SkillCorner matches with weak partial H/A labels; {len(split['validation'])} validation matches; {len(split['test'])} reserved test match(es). Expanded arms use the additional weak matches; base arms do not.",
        'Historical references and accuracy/spell changes:',json.dumps(references,indent=2),
        'External-selected candidate (separate from public winner):',json.dumps(summary['external_selected'],indent=2),
        'Public top three distinct full prediction files (decoder settings can share a trained model):']
    for i,row in enumerate(unique[:3],1):
        text.append(json.dumps(dict(rank=i,candidate=row['candidate'],decoder=row['decoder'],external=row['external'],
            public=row['public'],identical_prediction_aliases=row['equivalent_full_predictions']),indent=2))
    text += ['Best-public frozen recipes subject to external improvement and fewer missed possessions (diagnostic only; external-selected candidate unchanged):',
             json.dumps(joint,indent=2),
             'Duplicate-condition sensitivity: omit speed_missing for frozen external-selected recipe and two anchors; no reselection:',json.dumps(sensitivity,indent=2),
             'Operational recovery/resume audit and separate synthetic tests:',json.dumps(operational_recovery,indent=2),
             'Matched factorial effects:',json.dumps(effects,indent=2),
             'Held-out clean confirmation (read saved aggregates only; no test rerun):',json.dumps(heldout,indent=2),
             'Held-out changes:',json.dumps(heldout_changes,indent=2),
             'Limitations and regressions:']+limits
    for ref in references:
        for domain,change in ref['external_selected_change'].items():
            regressions={key:value for key,value in change.items() if key in METRICS and value<0}
            if regressions:text.append(f"External-selected candidate regressions versus weight{ref['missing_weight']} in {domain}: "+json.dumps(regressions))
    text += ['Every external recipe, per-match and per-scenario metrics: external_all_recipes.csv, external_per_match.csv, external_per_match_scenario.csv.',
        'All public recipes: ../public/leaderboard.csv. De-duplicated ranking: public_unique_predictions.csv; aliases: public_unique_predictions.json.',
        'Joint external/public diagnostic leaders, including accuracy and false-possession changes: joint_external_public_diagnostics.csv.',
        'Diagnostic five-condition mean/per-match sensitivity, preserving original selection: five_condition_sensitivity.csv.',
        'Training completion/epochs: training_runs.csv. Exact commands:',commands]
    (out/'REPORT.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    audit=out/'verification.json'
    paths={p.relative_to(ROOT).as_posix() for p in dest.rglob('*') if p.is_file()};paths.add(audit.relative_to(ROOT).as_posix())
    check=subprocess.run(['git','check-ignore','--no-index','-z','--stdin'],cwd=ROOT,input='\0'.join(sorted(paths))+'\0',capture_output=True,text=True,check=True)
    assert set(check.stdout.rstrip('\0').split('\0'))==paths
    assert not subprocess.check_output(['git','ls-files','--',str(dest.relative_to(ROOT))],cwd=ROOT)
    write(audit,dict(all_generated_files_ignored=True,tracked_generated_files=0,preserved_hashes_verified=True,
        checkpoint_hashes_verified=True,public_results_complete=True,only_aggregate_results_read=True,
        prediction_files_hashed_without_reading_rows=True,no_training_tuning_or_selection_change=True,source_sha256=sha(__file__)))
    print(json.dumps(dict(report=str(out/'REPORT.txt'),counts=counts,
        external_selected=dict(candidate=selected['candidate'],external_spellF1=selected['external']['spellF1'],public_spellF1=selected['public']['spellF1']),
        public_top_three=[dict(candidate=row['candidate'],decoder=row['decoder'].get('id'),external_spellF1=row['external']['spellF1'],public_spellF1=row['public']['spellF1']) for row in unique[:3]],verified=True)))
    return True


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dest',type=Path,default=DEFAULT_DEST)
    raise SystemExit(0 if run(parser.parse_args().dest) else 2)
