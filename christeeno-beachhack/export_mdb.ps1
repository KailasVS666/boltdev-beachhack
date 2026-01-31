$mdbPath = "c:\Users\sharj\Desktop\boltdev-beachhack\christeeno-beachhack\avall.mdb"
$outputDir = "c:\Users\sharj\Desktop\boltdev-beachhack\christeeno-beachhack\csv_output"

if (!(Test-Path $outputDir)) {
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
}

$connStr = "Provider=Microsoft.ACE.OLEDB.12.0;Data Source=$mdbPath;"
$conn = New-Object System.Data.OleDb.OleDbConnection($connStr)

try {
    $conn.Open()
    Write-Host "Connected to MDB successfully."

    # Get Tables
    $schema = $conn.GetOleDbSchemaTable([System.Data.OleDb.OleDbSchemaGuid]::Tables, $null)

    foreach ($row in $schema.Rows) {
        if ($row["TABLE_TYPE"] -eq "TABLE") {
            $tableName = $row["TABLE_NAME"]
            Write-Host "Processing Table: $tableName..."
            
            try {
                $cmd = New-Object System.Data.OleDb.OleDbCommand("SELECT * FROM [$tableName]", $conn)
                $adapter = New-Object System.Data.OleDb.OleDbDataAdapter($cmd)
                $dt = New-Object System.Data.DataTable
                $recordCount = $adapter.Fill($dt)
                
                Write-Host "  -> Loaded $recordCount rows."
                
                if ($recordCount -gt 0) {
                    $csvPath = Join-Path $outputDir "$tableName.csv"
                    # Use Export-Csv. Note: This can be slow for very large tables but is robust.
                    $dt | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
                    Write-Host "  -> Try Saved to $csvPath"
                }
                
                $dt.Dispose()
            } catch {
                Write-Error "Failed to export table $tableName : $_"
            }
        }
    }
} catch {
    Write-Error "Detailed Error: $_"
} finally {
    if ($conn.State -eq 'Open') { $conn.Close() }
}
