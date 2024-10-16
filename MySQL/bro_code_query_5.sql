# Video 7

# Creates a safepoint, this way, only be using COMMIT the safepoint changes

-- SET AUTOCOMMIT = OFF;

# Bypasses the error 1175 in the current session
-- SET SQL_SAFE_UPDATES = 0;

/*
DELETE FROM empregados;
SELECT * FROM empregados;


# Rollback Changes

ROLLBACK;

*/

SELECT * FROM empregados;