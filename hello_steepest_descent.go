/*
 * Иллюстрация метода наискорейшего спуска
 * 
 * go run hello_steepest_descent.go
 */
package main

import(
	"fmt"
	"math"
)

/*
	
 __
 \/F - оператор набла nabla F
 или другое обозначение grad F


  Градиентный спуск
  Алгоритм
	1.Задают начальное приближение и точность расчёта vec x[0],epsilon 
	2.Рассчитывают vec x[k+1]=vec x[k] - lambda[k] * nabla F(vec x[k]), 
	  где lambda[k] = argmin_lambda F(vec x[k] - lambda * nabla F(vec x[k]))
	3.Проверяют условие остановки:
	  Если |vec x[k+1] - vec x[k]| > epsilon 
	    или |F(vec x[k+1]) - F(vec x[k])| > epsilon  
		или |nabla F(vec x[k+1])| > epsilon (выбирают одно из условий), 
		то k = k + 1 и переход к пункту 2.
	  Иначе vec x = vec x[k+1] и останов.

	[https://ru.wikipedia.org/wiki/%D0%93%D1%80%D0%B0%D0%B4%D0%B8%D0%B5%D0%BD%D1%82%D0%BD%D1%8B%D0%B9_%D1%81%D0%BF%D1%83%D1%81%D0%BA]

  Алгоритм метода наискорейшего спуска
	Шаг 1. Задать x[0], epsilon1 > 0,epsilon2 > 0, предельное число итераций М. 
			Найти градиент функции в произвольной точке. Определить частные производные функции f(x)
			grad f(x) = [ df(x)/x1,...,df(x)/dxn ]T(транспонированный вектор)
	Шаг 2. Положить k = 0.
	Шаг 3. Вычислить grad f(x[k]).
	Шаг 4. Проверить выполнение критерия окончания  ||grad f(x[k])|| < epsilon1 :
		а) если критерий выполнен, то x[0] = x[k], останов;
		б) если критерий не выполнен, то перейти к шагу 5.
	Шаг 5. Проверить выполнение неравенства k ≥ M:
		а) если неравенство выполнено, то x[0] = x[k], останов;
		б) если нет, то перейти к шагу 6.
	Шаг 6. Вычислить величину шага lambda[k=0], из условия
		F(lambda[k]) = f(x[k] - lambda[k] * grad f(x[k])) -> min_lamda[k].
	Шаг 7. Вычислить x[k+1] = x[k] - lambda[k] * grad f(x[k]).
	Шаг 8. Проверить выполнение условий ||x[k+1] - x[k]|| < epsilon2, ||f(x[k+1]) - f(x[k])|| < epsilon2:
		а) если оба условия выполнены при текущем значении k и k = k - 1, то расчет окончен, x[0] = x[k+1];
		б) если хотя бы одно из условий не выполнено, то положить k = k + 1 и перейти к шагу 3.

	[https://studfiles.net/preview/1874323/page:2/]
*/

//
// main driver
//
func main() {
	fmt.Println("Hello steepest descent")
	//TODO: эти значения можно менять и наблюдать результат
	var x float64 = 34.0
	var y float64 = 12.0
	var epsilon = 0.01
	SteepestDescent(x, y, epsilon)
}

// Собственно здесь записывается наша функция
func f(x,y float64) float64 {
	//TODO: здеась можно задать исследуемею функцию или поиграться с этой
	//NOTE: если поменяете функцию не забудьте про частные производные
	return x*x + 2.0*x*y + 3.0*y*y - 2.0*x -3.0*y
}

//
// операция grad или nabla
//
// Это первая производная по dx
// частная производная по х
func f_dx(x, y float64) float64 {
	return 2.0*x + 2.0*y - 2.0
}
// Это первая производная по dy
// частная производная по y
func f_dy(x, y float64) float64 {
	return 2.0*x + 6.0*y - 3.0
}


// Это функция g(x[k]) в методе наискорейшего (градиентного) спуска
// применяемая для нахождения шага lambda[k] как минимума функции g(x[k])
// на отрезке(здес a=-10000, b=100000) (здесь методом дихотомии)
func g(x, y, lambda float64) float64 {
	return f(x - lambda * f_dx(x, y), y - lambda * f_dy(x, y))
}


// двумерная норма
// условие остановки
func norma(x, y float64) float64 {
	return math.Sqrt(x*x + y*y)
}


// Метод половинного деления для нахождения минимума в градиентном спуске
// dichotomia — разделение на две части
func dichotomia(a0, b0, epsilon, x, y float64) float64 {
	// Номер шага
	var k int
	// Отклонени от середины отрезка влево, вправо
	var left float64
	var right float64
	// Величина на которую мы отклонимся от середины отрезка
	var deviation float64 = 0.5 * epsilon
	// Точка минимума
	var x_min float64
	// Отрезок локализации минимума
	var ak float64 = a0
	var bk float64 = b0

	//	Шаг 3. Вычислить grad f(x[k]).
	//	Шаг 4. Проверить выполнение критерия окончания  ||grad f(x[k])|| < epsilon1 :
	//		а) если критерий выполнен, то x[0] = x[k], останов;
	//		б) если критерий не выполнен, то перейти к шагу 5.
	// Пока длина отрезка больше заданной точности
	for k = 0; (bk - ak) >= epsilon; k++ {
			// Берем середину (ну почти середину - +\- некоторое отклонение 
			// в частности у нас deviation = 0.5 * epsilon)
			left = (ak + bk - deviation) / 2.0
			right = (ak + bk + deviation) / 2.0
			// Шаг 3. Вычислить grad f(x[k])
			// Проверяем в какую часть попадает точка минимума 
			// слева от разбиения или справа и выбираем соответствующую точку
			if g(x, y, left) <= g(x, y, right) {
			// Теперь правая граница отрезка локализации равна right
					bk = right;
			} else {
			// Теперь левая граница отрезка локализации равна left
					ak = left;
			}
	}

	// minimum point
	x_min = (ak + bk) / 2.0

	return x_min;
}



// Метод наискорейшего спуска(пример алгоритма)
// the method of steepest descent or stationary-phase method or saddle-point method
func SteepestDescent(x0, y0, epsilon float64) float64 {
	const MAXITERATIONS = 1000
	var k int
	var x_cur float64
	var y_cur float64
	var x_next float64
	var y_next float64
	var lambda float64
	// П.1. Задают начальное приближение и точность расчёта vec x[0], epsilon
	//	Шаг 1. Задать x[0], epsilon1 > 0, epsilon2 > 0, предельное число итераций М. 
	//		Найти градиент функции в произвольной точке. Определить частные производные функции f(x):
	//		grad f(x) = [ df(x)/x1,...,df(x)/dxn ]T(транспонированный вектор)
  // Начальное приближение u[0]
	x_cur = x0
	y_cur = y0

	fmt.Println("## Исходная точка:")
	fmt.Printf("  x[0]: (%f, %f)\n", x_cur, y_cur)

	fmt.Println()
	fmt.Println("## Приближения:")
	//Шаг 2. Положить k = 0
	stop := false
	for k = 0; k < MAXITERATIONS && !stop ; k++ {
		// П.2. Рассчитывают vec x[k+1] = vec x[k] - lambda[k] * nabla F(vec x[k]), 
		// где lambda[k]= argmin_lambda F(vec x[k] - lambda * nabla F(vec x[k]))
		//	Шаг 3. Вычислить grad f(x[k]).
		//
		//	Шаг 4. Проверить выполнение критерия окончания  ||grad f(x[k])|| < epsilon1 :
		//		а) если критерий выполнен, то x[0] = x[k], останов;
		//		б) если критерий не выполнен, то перейти к шагу 5.
		//
		//	Шаг 5. Проверить выполнение неравенства k ≥ M:
		//		а) если неравенство выполнено, то x[0] = x[k], останов;
		//		б) если нет, то перейти к шагу 6
		//
		//	Шаг 6. Вычислить величину шага lambda[k0](на начальном шаге, k = 0) из условия
		//		F(lambda[k]) = f(x[k] - lambda[k] * grad f(x[k])) -> min lambda[k]
		//
		argmin := dichotomia
		// Находим lambda[k] как минимум функции g(x[k]) на отрезке -10000,100000
		lambda = argmin(-10000, 100000, epsilon, x_cur, y_cur);
		// Вычисляем u[k] новое прилижение
		// 	Шаг 7. Вычислить x[k+1] = x[k] - lambda[k] * grad f(x[k])
		x_next = x_cur - lambda * f_dx(x_cur, y_cur)
		y_next = y_cur - lambda * f_dy(x_cur, y_cur)

		fmt.Println()
		fmt.Printf("  x[%d]: (%.2f, %.2f)\n", k+1, x_next, y_next)
		fmt.Printf("  f(%.2f, %.2f) = %.2f\n", x_next, y_next, f(x_next, y_next))

		// П.3. Проверяют условие остановки:
		// Если |vec x[k+1] - vec x[k]| > epsilon 
		//  или |F(vec x[k+1]) - F(vec x[k])| > epsilon  
		//  или |nabla F(vec x[k+1])| > epsilon (выбирают одно из условий), 
		// то k = k + 1 и переход к П.2.
		// Иначе vec x = vec x[k+1] и останов.
		// 	Шаг 8. Проверить выполнение условий ||x[k+1] - x[k]|| < epsilon2, ||f(x[k+1]) - f(x[k])|| < epsilon2:
		//		а) если оба условия выполнены при текущем значении k и k = k - 1, то расчет окончен, x[0] = x[k+1], останов;
		//		б) если хотя бы одно из условий не выполнено, то положить k = k + 1 и перейти к шагу 3
		if k > 1 {
			// Проверяем условие остановки
			if norma(x_next - x_cur, y_next - y_cur) < epsilon {
				// останов
				stop = true
			}
		}

		x_cur = x_next
		y_cur = y_next
	}

	fmt.Println()
	fmt.Println("## Точка минимума epsilon:", epsilon)
	fmt.Printf("   f(%.2f , %.2f) = %.2f\n", x_cur, y_cur, f(x_cur, y_cur))

	// получить минимум функции
	return f(x_cur, y_cur)
}

