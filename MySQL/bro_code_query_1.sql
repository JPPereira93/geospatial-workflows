USE myDB_practice;

/*

# Video 3

empregados
CREATE TABLE empregados(
	id_empregados int,
    nome_proprio VARCHAR(255),
    apelido VARCHAR(255),
    salario_hora DECIMAL(5, 2),
    data_inicio DATE

);



-- Select all the table
-- SELECT * FROM empregados;

-- Drop a table
-- DROP TABLE empregados;

ALTER TABLE empregados
	ADD numero_telm VARCHAR(255);
    


ALTER TABLE empregados
	RENAME COLUMN numero_telm to email;
    


ALTER TABLE empregados
MODIFY COLUMN email VARCHAR(111);

ALTER TABLE empregados
MODIFY COLUMN email VARCHAR(111)
AFTER apelido -- Vem depois do apelido

*/

ALTER TABLE empregados
MODIFY COLUMN email VARCHAR(111)
FIRST; -- A coluna selecionada em cima vem primeiro