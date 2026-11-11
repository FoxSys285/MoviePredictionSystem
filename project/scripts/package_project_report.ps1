param([string]$StagingDirectory)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$taskRepository = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $StagingDirectory) { $StagingDirectory = Join-Path $taskRepository '.codex-artifacts\report' }
$taskStaging = (Resolve-Path -LiteralPath $StagingDirectory).Path
$taskPackage = Join-Path $taskStaging 'Nhom05_DoAn_Final.package'
$taskOriginal = Join-Path $taskStaging 'Nhom05_DoAn_BanGoc.docm'
$taskOutput = Join-Path $taskRepository 'Nhom05_DoAn.docm'
$taskPdf = Join-Path $taskRepository 'Nhom05_DoAn.pdf'
$taskArchive = Join-Path $taskRepository 'Nhom05_DoAn_HoanThien.zip'
$taskNamespace = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

function Read-ZipText($Archive, [string]$Name) {
    $taskReader = New-Object IO.StreamReader($Archive.GetEntry($Name).Open())
    try { return $taskReader.ReadToEnd() } finally { $taskReader.Dispose() }
}
function Assert-Condition([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Get-NormalizedText($Node, $NamespaceManager) {
    $taskText = ($Node.SelectNodes('.//w:t', $NamespaceManager) | ForEach-Object { $_.InnerText }) -join ''
    return ([regex]::Replace($taskText, '\s+', ' ')).Trim()
}

$taskInput = [IO.Compression.ZipFile]::OpenRead($taskPackage)
$taskBackup = [IO.Compression.ZipFile]::OpenRead($taskOriginal)
try {
    $taskBadParts = @($taskInput.Entries | Where-Object { $_.FullName -match '(?i)vba|customUI/|customXml/|embeddings/|activeX/' })
    Assert-Condition ($taskBadParts.Count -eq 0) 'Active or embedded executable parts found in report package.'
    foreach ($taskRel in @($taskInput.Entries | Where-Object { $_.FullName.EndsWith('.rels') })) {
        $taskRelText = Read-ZipText $taskInput $taskRel.FullName
        Assert-Condition ($taskRelText -notmatch '(?i)vbaProject|attachedTemplate|oleObject|ui/extensibility|/activeX') 'Unexpected active relationship.'
    }
    [xml]$taskDocument = Read-ZipText $taskInput 'word/document.xml'
    $taskNs = New-Object Xml.XmlNamespaceManager($taskDocument.NameTable)
    $taskNs.AddNamespace('w', $taskNamespace)
    $taskAllText = Get-NormalizedText $taskDocument $taskNs
    $taskCodes = ($taskDocument.SelectNodes('//w:instrText', $taskNs) | ForEach-Object { $_.InnerText }) -join ' '
    Assert-Condition ($taskCodes -notmatch '(?i)\bDDE(AUTO)?\b|\bINCLUDETEXT\b|\bINCLUDEPICTURE\b') 'Unexpected active field.'
    $taskTocCount = ([regex]::Matches($taskCodes, '\bTOC\b')).Count
    Assert-Condition ($taskTocCount -eq 3) 'Expected three automatic indexes.'
    $taskSections = $taskDocument.SelectNodes('//w:sectPr', $taskNs)
    Assert-Condition ($taskSections.Count -eq 3) 'Expected cover/front/body sections.'
    foreach ($taskSection in $taskSections) {
        $taskSize = $taskSection.SelectSingleNode('w:pgSz', $taskNs)
        Assert-Condition ([Math]::Abs([int]$taskSize.GetAttribute('w', $taskNamespace) - 11906) -le 2) 'Page width must be A4.'
        Assert-Condition ([Math]::Abs([int]$taskSize.GetAttribute('h', $taskNamespace) - 16838) -le 2) 'Page height must be A4.'
        $taskMargins = $taskSection.SelectSingleNode('w:pgMar', $taskNs)
        foreach ($taskEdge in @('left', 'right', 'top', 'bottom')) {
            $taskExpected = if ($taskEdge -eq 'left') { 1984 } else { 1417 }
            Assert-Condition ([Math]::Abs([int]$taskMargins.GetAttribute($taskEdge, $taskNamespace) - $taskExpected) -le 2) ('Unexpected margin: ' + $taskEdge)
        }
    }
    Assert-Condition ($taskSections[0].SelectNodes('w:footerReference', $taskNs).Count -eq 0) 'Cover pages must have no footer page number.'
    Assert-Condition ($taskSections[1].SelectSingleNode('w:pgNumType', $taskNs).GetAttribute('fmt', $taskNamespace) -eq 'lowerRoman') 'Front matter must use Roman numbers.'
    Assert-Condition ($taskSections[2].SelectSingleNode('w:pgNumType', $taskNs).GetAttribute('start', $taskNamespace) -eq '1') 'Body page number must restart at 1.'
    foreach ($taskHeading in @('CHƯƠNG 1: TỔNG QUAN ĐỀ TÀI', 'CHƯƠNG 2: CƠ SỞ LÝ THUYẾT', 'CHƯƠNG 3: MÔI TRƯỜNG THỰC NGHIỆM', 'CHƯƠNG 4: THỰC NGHIỆM VÀ SO SÁNH', 'KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN')) {
        Assert-Condition ($taskAllText.Contains($taskHeading)) ('Missing heading: ' + $taskHeading)
    }
    [xml]$taskOriginalDocument = Read-ZipText $taskBackup 'word/document.xml'
    $taskOriginalNs = New-Object Xml.XmlNamespaceManager($taskOriginalDocument.NameTable)
    $taskOriginalNs.AddNamespace('w', $taskNamespace)
    $taskCoverTexts = @()
    $taskOriginalBody = $taskOriginalDocument.SelectSingleNode('//w:body', $taskOriginalNs)
    for ($taskIndex = 0; $taskIndex -lt 48; $taskIndex++) {
        $taskCoverTexts += Get-NormalizedText $taskOriginalBody.ChildNodes[$taskIndex] $taskOriginalNs
    }
    $taskCoverText = [regex]::Replace(($taskCoverTexts -join ''), '\s+', '')
    Assert-Condition (([regex]::Replace($taskAllText, '\s+', '')).StartsWith($taskCoverText)) 'Cover text differs from the original two pages.'
    [xml]$taskStyles = Read-ZipText $taskInput 'word/styles.xml'
    $taskStyleNs = New-Object Xml.XmlNamespaceManager($taskStyles.NameTable)
    $taskStyleNs.AddNamespace('w', $taskNamespace)
    foreach ($taskStyleCheck in @(@('ReportBody', 26), @('ReportChapter', 36), @('ReportHeading1', 28), @('ReportHeading2', 26), @('ReportTableText', 24))) {
        $taskStyle = $taskStyles.SelectSingleNode(('//w:style[@w:styleId="' + $taskStyleCheck[0] + '"]'), $taskStyleNs)
        Assert-Condition ($null -ne $taskStyle) 'Required formatting style missing.'
        Assert-Condition ($taskStyle.SelectSingleNode('w:rPr/w:sz', $taskStyleNs).GetAttribute('val', $taskNamespace) -eq [string]$taskStyleCheck[1]) 'Unexpected font size.'
        Assert-Condition ($taskStyle.SelectSingleNode('w:rPr/w:rFonts', $taskStyleNs).GetAttribute('ascii', $taskNamespace) -eq 'Times New Roman') 'Unexpected font family.'
        Assert-Condition ($taskStyle.SelectSingleNode('w:pPr/w:spacing', $taskStyleNs).GetAttribute('line', $taskNamespace) -eq '312') 'Expected 1.3 line spacing.'
    }
    $taskFigureCount = $taskDocument.SelectNodes('//w:p[w:pPr/w:pStyle[@w:val="ReportFigureCaption"]]', $taskNs).Count
    $taskTableCaptionCount = $taskDocument.SelectNodes('//w:p[w:pPr/w:pStyle[@w:val="ReportTableCaption"]]', $taskNs).Count
    Assert-Condition ($taskFigureCount -eq 23 -and $taskTableCaptionCount -eq 35) 'Figure/table inventory mismatch.'
    $taskCompleted = Join-Path $taskStaging 'Nhom05_DoAn_Completed.docm'
    $taskOutputStream = [IO.File]::Create($taskCompleted)
    $taskDocm = New-Object IO.Compression.ZipArchive($taskOutputStream, [IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($taskEntry in $taskInput.Entries) {
            $taskNewEntry = $taskDocm.CreateEntry($taskEntry.FullName, [IO.Compression.CompressionLevel]::Optimal)
            $taskNewStream = $taskNewEntry.Open()
            try {
                if ($taskEntry.FullName -eq '[Content_Types].xml') {
                    $taskTypesText = (Read-ZipText $taskInput $taskEntry.FullName).Replace('application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml', 'application/vnd.ms-word.document.macroEnabled.main+xml')
                    $taskBytes = [Text.Encoding]::UTF8.GetBytes($taskTypesText)
                    $taskNewStream.Write($taskBytes, 0, $taskBytes.Length)
                } else {
                    $taskOldStream = $taskEntry.Open()
                    try { $taskOldStream.CopyTo($taskNewStream) } finally { $taskOldStream.Dispose() }
                }
            } finally { $taskNewStream.Dispose() }
        }
    } finally { $taskDocm.Dispose(); $taskOutputStream.Dispose() }
} finally { $taskInput.Dispose(); $taskBackup.Dispose() }

$taskOriginalHash = (Get-FileHash -LiteralPath $taskOriginal -Algorithm SHA256).Hash
if (Test-Path -LiteralPath $taskOutput) {
    $taskExistingHash = (Get-FileHash -LiteralPath $taskOutput -Algorithm SHA256).Hash
    $taskCompletedHash = (Get-FileHash -LiteralPath $taskCompleted -Algorithm SHA256).Hash
    Assert-Condition ($taskExistingHash -eq $taskOriginalHash -or $taskExistingHash -eq $taskCompletedHash) 'The original changed after the backup. Review the cover edits before replacing it.'
}
Copy-Item -LiteralPath $taskCompleted -Destination $taskOutput -Force
Copy-Item -LiteralPath (Join-Path $taskStaging 'Nhom05_DoAn_HoanThien.pdf') -Destination $taskPdf -Force

$taskPointer = Get-Content -LiteralPath (Join-Path $taskRepository 'project/artifacts/report-latest.json') -Encoding UTF8 -Raw | ConvertFrom-Json
$taskGuide = @"
BÁO CÁO ĐỒ ÁN NHÓM 05

Nhom05_DoAn.docx: bản Word có thể chỉnh sửa, không chứa VBA/ribbon tự chạy.
Nhom05_DoAn.pdf: bản xem/in đã kiểm tra, 59 trang, 23 hình và 35 bảng đánh số.
Hai trang bìa giữ nguyên để nhóm tự điền thông tin.
Số liệu: 49.552 ratings của mẫu 300 người dùng; không phải benchmark toàn bộ 25M.
Run ID: $($taskPointer.run_id). Chia thời gian 70/15/15; CV 5 fold thời gian.
Ket_qua_thuc_nghiem.csv: log số liệu gốc dùng để lập bảng và biểu đồ.

Không bật macro khi mở tài liệu. Bản nguồn được cung cấp có mã tự chạy; phần này đã bị loại bỏ khỏi bản bàn giao.
Windows trên máy thực hiện cũng đã chặn một số chương trình bằng cảnh báo virus. Bản bàn giao không xác nhận máy tính đã được làm sạch.
"@
$taskDeliveryStream = [IO.File]::Create($taskArchive)
$taskDelivery = New-Object IO.Compression.ZipArchive($taskDeliveryStream, [IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($taskPair in @(@($taskPackage, 'Nhom05_DoAn.docx'), @($taskPdf, 'Nhom05_DoAn.pdf'), @((Join-Path $taskPointer.directory 'metrics.csv'), 'Ket_qua_thuc_nghiem.csv'))) {
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($taskDelivery, $taskPair[0], $taskPair[1], [IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }
    $taskGuideEntry = $taskDelivery.CreateEntry('HUONG_DAN.txt')
    $taskGuideWriter = New-Object IO.StreamWriter($taskGuideEntry.Open(), (New-Object Text.UTF8Encoding($false)))
    try { $taskGuideWriter.Write($taskGuide) } finally { $taskGuideWriter.Dispose() }
} finally { $taskDelivery.Dispose(); $taskDeliveryStream.Dispose() }

$taskCheck = [IO.Compression.ZipFile]::OpenRead($taskOutput)
try {
    Assert-Condition (@($taskCheck.Entries | Where-Object { $_.FullName -match '(?i)vba|customUI/|customXml/|embeddings/|activeX/' }).Count -eq 0) 'Delivered DOCM has unexpected active parts.'
} finally { $taskCheck.Dispose() }
$taskRender = Get-Content -LiteralPath (Join-Path $taskStaging 'libreoffice-validation.json') -Encoding UTF8 -Raw | ConvertFrom-Json
$taskValidation = [ordered]@{
    run_id = $taskPointer.run_id; pages = $taskRender.statistics.PageCount
    numbered_figures = $taskFigureCount; numbered_tables = $taskTableCaptionCount
    automatic_indexes = $taskTocCount; covers_preserved = $true
    a4_margins_font_spacing_verified = $true; active_parts_removed = $true
    files = @(Get-FileHash -LiteralPath $taskOutput, $taskPdf, $taskArchive -Algorithm SHA256 | Select-Object Path, Hash)
}
$taskValidation | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskRepository 'project/docs/report-validation.json') -Encoding UTF8
$taskValidation | ConvertTo-Json -Depth 5
