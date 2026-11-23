param([string]$InputDocument,[string]$OutFile)
$ErrorActionPreference='Stop'
$w=New-Object -ComObject Word.Application
$w.Visible=$false
$w.DisplayAlerts=0
try {
  $d=$w.Documents.Open([IO.Path]::GetFullPath($InputDocument),$false,$true)
  $d.Repaginate()
  $items=foreach($par in $d.Paragraphs){
    $r=$par.Range
    if($r.Information(12)){continue}
    $s=$r.Text.TrimEnd([char]13)
    if($s.StartsWith('Referencias bibliogr')){break}
    if([string]::IsNullOrWhiteSpace($s)){continue}
    $n=$r.ComputeStatistics(1)
    $cuts=@()
    if($n -ge 4){
      $candidates=@()
      foreach($m in [regex]::Matches($s,'[.;](?=\s)')){
        $pre=$s.Substring(0,$m.Index+1)
        if($pre -match '(et al\.|s\. f\.|\bs\.|\bSens\.|\bEspec\.)$' -or ($pre.LastIndexOf('(') -gt $pre.LastIndexOf(')'))){continue}
        $part=$r.Duplicate
        $part.End=$r.Start+$m.Index+1
        $candidates+=,[pscustomobject]@{pos=$m.Index+1;line=$part.ComputeStatistics(1)}
      }
      if($n -lt 8){
        $hit=$candidates | Where-Object { $_.line -ge $n*.30 -and $_.line -le $n*.70+0.5 } | Sort-Object { [Math]::Abs($_.pos-$s.Length/2) } | Select-Object -First 1
        if($hit){$cuts+=$hit.pos}
      }else{
        $startLine=0
        while($n-$startLine -gt 8){
          $hit=$candidates | Where-Object {$_.line -gt $startLine -and $_.line -le $startLine+8} | Select-Object -Last 1
          if(!$hit){$hit=$candidates | Where-Object {$_.line -gt $startLine} | Select-Object -First 1}
          if(!$hit){break}
          $cuts+=$hit.pos
          $startLine=$hit.line
        }
      }
    }
    [pscustomobject]@{text=$s;lines=$n;cut=$cuts}
  }
  $items | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 -LiteralPath $OutFile
} finally {if($d){$d.Close(0)};$w.Quit()}
