Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Select-Object ProcessId, CreationDate, CommandLine |
  Format-List |
  Out-File -FilePath py_procs.txt -Encoding utf8
Get-Content py_procs.txt
