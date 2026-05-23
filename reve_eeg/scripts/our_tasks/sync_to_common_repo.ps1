param(
    [string]$SourceRepo = "\\10.16.93.90\dataset3\nzh\eeg_FM\reve_eeg",
    [string]$TargetRepo = "\\10.16.93.90\dataset3\nzh\eeg_FM\ML_common_project_topic",
    [string]$TargetSubdir = "reve_eeg"
)

$ErrorActionPreference = "Stop"

function Test-ShouldSkipFile {
    param(
        [string]$RelativePath,
        [string]$FileName
    )

    if ($FileName -like "prompt_*.md") {
        return $true
    }
    if ($FileName -like "*.pyc" -or $FileName -like "*.pyo") {
        return $true
    }
    if ($FileName -like "*.pth" -or $FileName -like "*.pt") {
        return $true
    }

    $parts = $RelativePath -split "[\\/]"
    foreach ($part in $parts) {
        if ($part -in @(".git", "checkpoints", "__pycache__", ".ipynb_checkpoints")) {
            return $true
        }
    }

    return $false
}

function Get-RelativePathSafe {
    param(
        [string]$BasePath,
        [string]$ChildPath
    )

    $baseItem = Get-Item -LiteralPath $BasePath
    $childItem = Get-Item -LiteralPath $ChildPath

    $baseFull = $baseItem.FullName.TrimEnd('\')
    $childFull = $childItem.FullName

    $baseUri = New-Object System.Uri(($baseFull + '\'))
    $childUri = New-Object System.Uri($childFull)
    $relativeUri = $baseUri.MakeRelativeUri($childUri)
    return [System.Uri]::UnescapeDataString($relativeUri.ToString()).Replace('/', '\')
}

if (-not (Test-Path -LiteralPath $SourceRepo)) {
    throw "Source repo not found: $SourceRepo"
}

if (-not (Test-Path -LiteralPath $TargetRepo)) {
    throw "Target repo not found: $TargetRepo"
}

$targetRoot = Join-Path $TargetRepo $TargetSubdir
New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null

$sourceRootResolved = (Resolve-Path -LiteralPath $SourceRepo).Path
$files = Get-ChildItem -LiteralPath $SourceRepo -Recurse -File
$copied = 0
$skipped = 0

foreach ($file in $files) {
    $relative = Get-RelativePathSafe -BasePath $sourceRootResolved -ChildPath $file.FullName
    if (Test-ShouldSkipFile -RelativePath $relative -FileName $file.Name) {
        $skipped += 1
        continue
    }

    $targetPath = Join-Path $targetRoot $relative
    $targetDir = Split-Path -Parent $targetPath
    if (-not (Test-Path -LiteralPath $targetDir)) {
        New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    }

    Copy-Item -LiteralPath $file.FullName -Destination $targetPath -Force
    $copied += 1
}

Write-Host "[SYNC] source: $SourceRepo"
Write-Host "[SYNC] target: $targetRoot"
Write-Host "[SYNC] copied files: $copied"
Write-Host "[SYNC] skipped files: $skipped"
