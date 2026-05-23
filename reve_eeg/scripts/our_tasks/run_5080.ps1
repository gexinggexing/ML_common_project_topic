param(
    [string]$Preset = "smoke",
    [string]$Datasets = "SEED",
    [string]$Modes = "lp",
    [string]$Seeds = "42",
    [string]$PythonExe = "D:\app\conda_envs\brain-dl\python.exe",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location -LiteralPath $repoRoot

Write-Host "Output directory: outputs/course_project"
Write-Host "TensorBoard: tensorboard --logdir outputs/course_project --host 127.0.0.1 --port 6006"
Write-Host "Browser: http://127.0.0.1:6006"

& $PythonExe "scripts/our_tasks/run_course_project.py" `
    --resource 5080 `
    --preset $Preset `
    --datasets $Datasets `
    --modes $Modes `
    --seeds $Seeds `
    --python-exe $PythonExe `
    @ExtraArgs
