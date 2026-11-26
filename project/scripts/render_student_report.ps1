param([string]$ReportPath)
$ErrorActionPreference = 'Stop'
if (-not $ReportPath) {
    $ReportPath = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'BaoCao_MovieLens25M.docx'
}
$taskReport = (Resolve-Path -LiteralPath $ReportPath).Path
$taskPdf = [System.IO.Path]::ChangeExtension($taskReport, '.pdf')
$taskStats = [System.IO.Path]::ChangeExtension($taskReport, '.pagination.json')
$taskWord = $null
$taskDoc = $null
try {
    $taskWord = New-Object -ComObject Word.Application
    $taskWord.Visible = $false
    $taskWord.DisplayAlerts = 0
    $taskWord.AutomationSecurity = 3
    $taskDoc = $taskWord.Documents.Open($taskReport, $false, $false, $false)
    $taskDoc.TrackRevisions = $false
    for ($taskPass = 0; $taskPass -lt 3; $taskPass++) {
        $taskDoc.Fields.Update() | Out-Null
        for ($taskIndex = 1; $taskIndex -le $taskDoc.TablesOfContents.Count; $taskIndex++) {
            $taskDoc.TablesOfContents.Item($taskIndex).Update()
        }
        $taskDoc.Repaginate()
    }
    $taskDoc.Save()
    $taskDoc.ExportAsFixedFormat($taskPdf, 17)
    $taskPageCount = $taskDoc.ComputeStatistics(2)
    $taskSections = @()
    for ($taskSectionIndex = 1; $taskSectionIndex -le $taskDoc.Sections.Count; $taskSectionIndex++) {
        $taskSection = $taskDoc.Sections.Item($taskSectionIndex)
        $taskSections += [pscustomobject]@{
            section = $taskSectionIndex
            start_page = $taskSection.Range.Information(3)
            left_cm = $taskWord.PointsToCentimeters($taskSection.PageSetup.LeftMargin)
            right_cm = $taskWord.PointsToCentimeters($taskSection.PageSetup.RightMargin)
            top_cm = $taskWord.PointsToCentimeters($taskSection.PageSetup.TopMargin)
            bottom_cm = $taskWord.PointsToCentimeters($taskSection.PageSetup.BottomMargin)
        }
    }
    $taskTitles = @()
    foreach ($taskParagraph in $taskDoc.Paragraphs) {
        $taskText = $taskParagraph.Range.Text.Trim()
        if ($taskText -match '^CH.+ [1-5]:' -and $taskText.Length -lt 120) {
            $taskTitles += [pscustomobject]@{title = $taskText; page = $taskParagraph.Range.Information(3)}
        }
    }
    [pscustomobject]@{
        pages = $taskPageCount
        tables_of_contents = $taskDoc.TablesOfContents.Count
        tables = $taskDoc.Tables.Count
        sections = $taskSections
        chapter_pages = $taskTitles
        pdf = $taskPdf
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $taskStats -Encoding UTF8
    Write-Output ('Pages: ' + $taskPageCount)
    Write-Output ('PDF: ' + $taskPdf)
} finally {
    if ($null -ne $taskDoc) { $taskDoc.Close(0) }
    if ($null -ne $taskWord) { $taskWord.Quit() }
}
