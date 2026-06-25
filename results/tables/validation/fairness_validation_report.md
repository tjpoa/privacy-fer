# Fairness Validation Report

This report checks whether the comparison protocol uses consistent splits, consistent transformation application, and a validation-first tuning workflow.

## Summary

- PASS: 26
- WARN: 1
- FAIL: 0

## Checks

- **PASS** `train_baseline_distribution`: {"class_distribution": {"angry": 4289, "disgust": 4289, "fear": 4289, "happy": 4289, "neutral": 4289, "sad": 4289, "surprise": 4289}, "n_samples": 30023}
- **PASS** `train_baseline_clean_same_samples_as_baseline`: mode=none, intensity=0.0, baseline_n=30023, transformed_n=30023
- **PASS** `train_crop_context_removal_same_samples_as_baseline`: mode=crop, intensity=0.75, baseline_n=30023, transformed_n=30023
- **PASS** `train_blur_same_samples_as_baseline`: mode=blur, intensity=3.0, baseline_n=30023, transformed_n=30023
- **PASS** `train_mosaic_same_samples_as_baseline`: mode=mosaic, intensity=8.0, baseline_n=30023, transformed_n=30023
- **PASS** `train_canny_same_samples_as_baseline`: mode=edges, intensity=0.0, baseline_n=30023, transformed_n=30023
- **PASS** `train_noise_same_samples_as_baseline`: mode=noise, intensity=100.0, baseline_n=30023, transformed_n=30023
- **PASS** `val_baseline_distribution`: {"class_distribution": {"angry": 1072, "disgust": 1072, "fear": 1072, "happy": 1072, "neutral": 1072, "sad": 1072, "surprise": 1072}, "n_samples": 7504}
- **PASS** `val_baseline_clean_same_samples_as_baseline`: mode=none, intensity=0.0, baseline_n=7504, transformed_n=7504
- **PASS** `val_crop_context_removal_same_samples_as_baseline`: mode=crop, intensity=0.75, baseline_n=7504, transformed_n=7504
- **PASS** `val_blur_same_samples_as_baseline`: mode=blur, intensity=3.0, baseline_n=7504, transformed_n=7504
- **PASS** `val_mosaic_same_samples_as_baseline`: mode=mosaic, intensity=8.0, baseline_n=7504, transformed_n=7504
- **PASS** `val_canny_same_samples_as_baseline`: mode=edges, intensity=0.0, baseline_n=7504, transformed_n=7504
- **PASS** `val_noise_same_samples_as_baseline`: mode=noise, intensity=100.0, baseline_n=7504, transformed_n=7504
- **PASS** `test_baseline_distribution`: {"class_distribution": {"angry": 595, "disgust": 595, "fear": 595, "happy": 595, "neutral": 595, "sad": 595, "surprise": 595}, "n_samples": 4165}
- **PASS** `test_baseline_clean_same_samples_as_baseline`: mode=none, intensity=0.0, baseline_n=4165, transformed_n=4165
- **PASS** `test_crop_context_removal_same_samples_as_baseline`: mode=crop, intensity=0.75, baseline_n=4165, transformed_n=4165
- **PASS** `test_blur_same_samples_as_baseline`: mode=blur, intensity=3.0, baseline_n=4165, transformed_n=4165
- **PASS** `test_mosaic_same_samples_as_baseline`: mode=mosaic, intensity=8.0, baseline_n=4165, transformed_n=4165
- **PASS** `test_canny_same_samples_as_baseline`: mode=edges, intensity=0.0, baseline_n=4165, transformed_n=4165
- **PASS** `test_noise_same_samples_as_baseline`: mode=noise, intensity=100.0, baseline_n=4165, transformed_n=4165
- **PASS** `results_summary_exists`: C:\Users\Tiago\Documents\GitHub\privacy-fer\results\tables\final\results_summary.csv rows=17
- **PASS** `results_summary_required_columns`: all required columns present
- **PASS** `results_summary_transformation_coverage`: observed=baseline_clean,blur,canny,crop_context_removal,mosaic,noise; missing=
- **PASS** `privacy_parameter_selection_has_validation_columns`: results\tables\intermediate\deid_proxy_selection.csv; columns=group,run_name,model,privacy_mode,privacy_intensity,best_epoch,epochs,batch_size,num_workers,metrics_path,val_accuracy,val_precision_macro,val_recall_macro,val_f1_macro,val_precision_weighted,val_recall_weighted,val_f1_weighted,val_loss,test_accuracy,test_precision_macro,test_recall_macro,test_f1_macro,test_precision_weighted,test_recall_weighted,test_f1_weighted,test_loss,privacy_score,val_f1_drop,within_drop_limit
- **PASS** `final_deid_table_separates_validation_and_test_metrics`: results\tables\intermediate\deid_fixed_comparison.csv; rows=9
- **WARN** `test_set_not_used_for_tuning`: Code-level evidence supports validation-based selection: src/modeling/training.py selects best_epoch using validation macro-F1 and deid_proxy_selection.csv contains validation selection columns. Human notebook decisions cannot be fully proven from artifacts alone; report should state that parameter choices are based on validation metrics and test is reserved for final reporting.
