USE myDB_practice;

# Video 6

# Basically i add id_empregados column again with add autoincremental column and as a primary key

# Steps:

/*
CREATE TABLE empregados_backup AS SELECT * FROM empregados;

ALTER TABLE empregados DROP COLUMN id_empregados;

ALTER TABLE empregados
ADD id_empregados INT PRIMARY KEY AUTO_INCREMENT;




UPDATE empregados
SET salario_hora = 10.25,
	data_inicio = "1993-11-24"
    
# If no WHERE is given, the entire column is updated

WHERE id_empregados = 2;



UPDATE empregados
SET data_inicio = NULL
WHERE apelido = 2;

*/

# Delete a row from a table

DELETE from empregados
WHERE id_empregados = 2;