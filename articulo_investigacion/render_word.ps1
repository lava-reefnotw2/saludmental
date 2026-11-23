param([string]$InputDocument,[string]$OutputPdf)
$ErrorActionPreference='Stop'
$InputDocument=[System.IO.Path]::GetFullPath($InputDocument)
$OutputPdf=[System.IO.Path]::GetFullPath($OutputPdf)
$wordApp=$null
$wordDoc=$null
try {
    $wordApp=New-Object -ComObject Word.Application
    $wordApp.Visible=$false
    $wordApp.DisplayAlerts=0
    $wordDoc=$wordApp.Documents.Open($InputDocument,$false,$true)
    $wordDoc.Repaginate()
    $wordDoc.ExportAsFixedFormat($OutputPdf,17)
    Write-Output ('Paginas: '+$wordDoc.ComputeStatistics(2))
} finally {
    if ($null -ne $wordDoc) {$wordDoc.Close(0);[void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordDoc)}
    if ($null -ne $wordApp) {$wordApp.Quit();[void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordApp)}
}
